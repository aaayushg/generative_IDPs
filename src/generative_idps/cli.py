from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from .analysis import (
    compute_minimum_rmsd_distribution,
    extract_minimum_rmsd,
    plot_histogram_grid,
    run_kmeans_analysis,
)
from .constants import DEFAULT_ATOM_NAMES, DEFAULT_RANDOM_SEED
from .training import generate_coordinate_structures, train_coordinate_autoencoder, train_tabular_autoencoder


def _parse_atom_names(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_ATOM_NAMES)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return tuple(values)


def _parse_legacy_input(path: Union[str, Path]) -> Dict[str, str]:
    parameters: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, value = stripped.split(maxsplit=1)
            parameters[key] = value
    return parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generative-idps")
    subparsers = parser.add_subparsers(dest="command", required=True)

    coordinate = subparsers.add_parser("train-coordinate", help="Train an autoencoder from a multi-model PDB trajectory")
    coordinate.add_argument("--input-pdb", required=True)
    coordinate.add_argument("--output-dir", required=True)
    coordinate.add_argument("--latent-dim", required=True, type=int)
    coordinate.add_argument("--atom-names", default=",".join(DEFAULT_ATOM_NAMES))
    coordinate.add_argument("--test-size", type=int)
    coordinate.add_argument("--test-fraction", type=float, default=0.2)
    coordinate.add_argument("--split-mode", choices=["sequential", "random"], default="sequential")
    coordinate.add_argument("--epochs", type=int, default=100)
    coordinate.add_argument("--batch-size", type=int, default=40)
    coordinate.add_argument("--optimizer", default="adam")
    coordinate.add_argument("--hidden-dims", type=_parse_hidden_dims, default=(300, 50))
    coordinate.add_argument("--sample-count", type=int, default=100)
    coordinate.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    coordinate.add_argument("--label", default="protein")
    coordinate.add_argument("--quiet", action="store_true")

    generate = subparsers.add_parser("generate-coordinate", help="Sample new structures from a saved decoder")
    generate.add_argument("--training-dir", required=True)
    generate.add_argument("--sample-count", required=True, type=int)
    generate.add_argument("--output-pdb")
    generate.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    tabular = subparsers.add_parser("train-tabular", help="Train an autoencoder on a numeric matrix")
    tabular.add_argument("--input-table", required=True)
    tabular.add_argument("--output-dir", required=True)
    tabular.add_argument("--latent-dim", required=True, type=int)
    tabular.add_argument("--test-size", type=int)
    tabular.add_argument("--test-fraction", type=float, default=0.2)
    tabular.add_argument("--split-mode", choices=["sequential", "random"], default="sequential")
    tabular.add_argument("--epochs", type=int, default=100)
    tabular.add_argument("--batch-size", type=int, default=40)
    tabular.add_argument("--optimizer", default="adam")
    tabular.add_argument("--hidden-dims", type=_parse_hidden_dims, default=(300, 50))
    tabular.add_argument("--sample-count", type=int, default=100)
    tabular.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    tabular.add_argument("--label", default="tabular")
    tabular.add_argument("--delimiter")
    tabular.add_argument("--quiet", action="store_true")

    cluster = subparsers.add_parser("cluster", help="Run K-means clustering on an encoded dataset")
    cluster.add_argument("--input", required=True)
    cluster.add_argument("--output-dir", required=True)
    cluster.add_argument("--clusters", required=True, type=int)
    cluster.add_argument("--columns", help="Comma-separated zero-based column indices")

    histogram = subparsers.add_parser("plot-histogram", help="Create a 3-panel histogram plot for latent datasets")
    histogram.add_argument("--training", required=True)
    histogram.add_argument("--test", required=True)
    histogram.add_argument("--generated", required=True)
    histogram.add_argument("--output", required=True)
    histogram.add_argument("--x-dim", type=int, default=0)
    histogram.add_argument("--y-dim", type=int, default=1)
    histogram.add_argument("--bins", type=int, default=40)

    rmsd = subparsers.add_parser("rmsd", help="Compare two trajectories with Kabsch-aligned RMSD")
    rmsd.add_argument("--reference-pdb", required=True)
    rmsd.add_argument("--candidate-pdb", required=True)
    rmsd.add_argument("--atom-names", default=",".join(DEFAULT_ATOM_NAMES))
    rmsd.add_argument("--unique-match", action="store_true")
    rmsd.add_argument("--output")

    minimum = subparsers.add_parser("extract-min-rmsd", help="Extract minimum RMSD per group from a whitespace table")
    minimum.add_argument("--input", required=True)
    minimum.add_argument("--output")
    minimum.add_argument("--group-column", type=int, default=0)
    minimum.add_argument("--value-column", type=int, default=2)

    legacy = subparsers.add_parser("train-legacy-coordinate", help="Run coordinate training from the original key-value input file")
    legacy.add_argument("legacy_input")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-coordinate":
        outputs = train_coordinate_autoencoder(
            input_pdb=args.input_pdb,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            atom_names=_parse_atom_names(args.atom_names),
            test_size=args.test_size,
            test_fraction=args.test_fraction,
            split_mode=args.split_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            hidden_dims=args.hidden_dims,
            sample_count=args.sample_count,
            seed=args.seed,
            label=args.label,
            verbose=0 if args.quiet else 1,
        )
        print(json.dumps({"output_dir": str(outputs.output_dir.resolve())}, indent=2))
        return 0

    if args.command == "generate-coordinate":
        output = generate_coordinate_structures(
            training_dir=args.training_dir,
            sample_count=args.sample_count,
            output_pdb=args.output_pdb,
            seed=args.seed,
        )
        print(json.dumps({"output_pdb": str(Path(output).resolve())}, indent=2))
        return 0

    if args.command == "train-tabular":
        outputs = train_tabular_autoencoder(
            input_table=args.input_table,
            output_dir=args.output_dir,
            latent_dim=args.latent_dim,
            test_size=args.test_size,
            test_fraction=args.test_fraction,
            split_mode=args.split_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            hidden_dims=args.hidden_dims,
            sample_count=args.sample_count,
            seed=args.seed,
            label=args.label,
            delimiter=args.delimiter,
            verbose=0 if args.quiet else 1,
        )
        print(json.dumps({"output_dir": str(outputs.output_dir.resolve())}, indent=2))
        return 0

    if args.command == "cluster":
        columns = None
        if args.columns:
            columns = [int(item.strip()) for item in args.columns.split(",") if item.strip()]
        result = run_kmeans_analysis(
            input_path=args.input,
            output_dir=args.output_dir,
            n_clusters=args.clusters,
            columns=columns,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "plot-histogram":
        output = plot_histogram_grid(
            training_path=args.training,
            test_path=args.test,
            generated_path=args.generated,
            output_path=args.output,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            bins=args.bins,
        )
        print(json.dumps({"output": str(Path(output).resolve())}, indent=2))
        return 0

    if args.command == "rmsd":
        result = compute_minimum_rmsd_distribution(
            reference_pdb=args.reference_pdb,
            candidate_pdb=args.candidate_pdb,
            atom_names=_parse_atom_names(args.atom_names),
            unique_match=args.unique_match,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "extract-min-rmsd":
        result = extract_minimum_rmsd(
            input_path=args.input,
            output_path=args.output,
            group_column=args.group_column,
            value_column=args.value_column,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "train-legacy-coordinate":
        params = _parse_legacy_input(args.legacy_input)
        outputs = train_coordinate_autoencoder(
            input_pdb=params["trajectory"],
            output_dir=params.get("out_folder", "outputs/legacy_run"),
            latent_dim=int(params["dimensions"]),
            test_size=int(params.get("test_size", 200)),
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 40)),
            optimizer=params.get("optimizer", "adam"),
            label=params.get("out_label", "protein"),
            verbose=1,
        )
        if "encoder_file" in params:
            encoder_destination = Path(params["encoder_file"])
            encoder_destination.parent.mkdir(parents=True, exist_ok=True)
            if outputs.encoder_path.resolve() != encoder_destination.resolve():
                shutil.copy2(outputs.encoder_path, encoder_destination)
        if "decoder_file" in params:
            decoder_destination = Path(params["decoder_file"])
            decoder_destination.parent.mkdir(parents=True, exist_ok=True)
            if outputs.decoder_path.resolve() != decoder_destination.resolve():
                shutil.copy2(outputs.decoder_path, decoder_destination)
        print(json.dumps({"output_dir": str(outputs.output_dir.resolve())}, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1
