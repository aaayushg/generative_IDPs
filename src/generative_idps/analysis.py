from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .constants import DEFAULT_ATOM_NAMES
from .trajectory import PDBTrajectory


def run_kmeans_analysis(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    n_clusters: int,
    columns: Optional[List[int]] = None,
    random_state: int = 123,
) -> dict:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, sep=r"\s+", header=None)
    selected = frame if columns is None else frame.iloc[:, columns]
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(selected)
    score = silhouette_score(selected, labels, metric="sqeuclidean")

    labels_path = output_dir / "cluster_labels.csv"
    centers_path = output_dir / "cluster_centers.csv"
    summary_path = output_dir / "clustering_summary.json"

    pd.DataFrame({"cluster": labels}).to_csv(labels_path, index=False)
    pd.DataFrame(model.cluster_centers_).to_csv(centers_path, index=False)
    summary = {
        "input_path": str(Path(input_path).resolve()),
        "columns": columns,
        "n_clusters": n_clusters,
        "silhouette_score": float(score),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_histogram_grid(
    training_path: Union[str, Path],
    test_path: Union[str, Path],
    generated_path: Union[str, Path],
    output_path: Union[str, Path],
    x_dim: int = 0,
    y_dim: int = 1,
    bins: int = 40,
) -> Path:
    import matplotlib.pyplot as plt

    datasets = [
        ("Training", np.loadtxt(training_path)),
        ("Test", np.loadtxt(test_path)),
        ("Generated", np.loadtxt(generated_path)),
    ]
    arrays = []
    for _, dataset in datasets:
        if dataset.ndim == 1:
            dataset = dataset.reshape(-1, 1)
        arrays.append(dataset)
    all_x = np.concatenate([dataset[:, x_dim] for dataset in arrays])
    all_y = np.concatenate([dataset[:, y_dim] for dataset in arrays])
    hist_range = np.array([(float(all_x.min()), float(all_x.max())), (float(all_y.min()), float(all_y.max()))])

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, label, dataset in zip(axes, [item[0] for item in datasets], arrays):
        axis.hist2d(dataset[:, x_dim], dataset[:, y_dim], bins=bins, range=hist_range)
        axis.set_title(label)
        axis.set_xlabel(f"Latent dimension {x_dim}")
        axis.set_ylabel(f"Latent dimension {y_dim}")
        axis.set_box_aspect(1)
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def kabsch_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    reference_centered = reference - np.mean(reference, axis=0)
    candidate_centered = candidate - np.mean(candidate, axis=0)
    covariance = candidate_centered.T @ reference_centered
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    rotation = u_matrix @ v_transpose
    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1
        rotation = u_matrix @ v_transpose
    rotated = candidate_centered @ rotation
    diff = rotated - reference_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_minimum_rmsd_distribution(
    reference_pdb: Union[str, Path],
    candidate_pdb: Union[str, Path],
    atom_names: Optional[List[str]] = None,
    unique_match: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    max_unique_matrix_size: int = 5_000_000,
) -> dict:
    atom_names = atom_names or list(DEFAULT_ATOM_NAMES)
    reference = PDBTrajectory.from_pdb(reference_pdb)
    candidate = PDBTrajectory.from_pdb(candidate_pdb)
    reference = reference.subset(atom_indices=reference.select_indices(atom_names))
    candidate = candidate.subset(atom_indices=candidate.select_indices(atom_names))
    if reference.n_atoms != candidate.n_atoms:
        raise ValueError("Reference and candidate selections must contain the same number of atoms")

    if unique_match:
        matrix_size = reference.n_frames * candidate.n_frames
        if matrix_size > max_unique_matrix_size:
            raise ValueError("Unique matching would allocate an excessively large RMSD matrix")
        matrix = np.zeros((candidate.n_frames, reference.n_frames), dtype=float)
        for cand_index in range(candidate.n_frames):
            for ref_index in range(reference.n_frames):
                matrix[cand_index, ref_index] = kabsch_rmsd(reference.coordinates[ref_index], candidate.coordinates[cand_index])
        cand_assignment, ref_assignment = linear_sum_assignment(matrix)
        minima = matrix[cand_assignment, ref_assignment]
    else:
        minima = np.zeros(candidate.n_frames, dtype=float)
        for cand_index in range(candidate.n_frames):
            rmsds = [kabsch_rmsd(reference_frame, candidate.coordinates[cand_index]) for reference_frame in reference.coordinates]
            minima[cand_index] = float(np.min(rmsds))

    if output_path is not None:
        np.savetxt(output_path, minima, fmt="%.8f")

    return {
        "count": int(minima.shape[0]),
        "mean": float(np.mean(minima)),
        "std": float(np.std(minima)),
        "min": float(np.min(minima)),
        "max": float(np.max(minima)),
    }


def extract_minimum_rmsd(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    group_column: int = 0,
    value_column: int = 2,
) -> dict:
    frame = pd.read_csv(input_path, sep=r"\s+", header=None)
    minima = frame.groupby(group_column)[value_column].min().sort_index()
    if output_path is not None:
        minima.to_csv(output_path, index=True, header=False)
    return {
        "count": int(minima.shape[0]),
        "mean": float(minima.mean()),
        "std": float(minima.std(ddof=0)),
        "min": float(minima.min()),
        "max": float(minima.max()),
    }
