from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .constants import DEFAULT_ATOM_NAMES, DEFAULT_RANDOM_SEED
from .trajectory import PDBTrajectory


@dataclass
class TrainingOutputs:
    output_dir: Path
    metadata_path: Path
    encoder_path: Path
    decoder_path: Path
    scaler_path: Path


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        return


def build_dense_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...] = (300, 50),
    activation: str = "relu",
    output_activation: str = "sigmoid",
):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Input

    encoder_layers = [Dense(units, activation=activation, name=f"encoder_dense_{index}") for index, units in enumerate(hidden_dims)]
    latent_layer = Dense(latent_dim, activation=activation, name="latent")
    decoder_layers = [
        Dense(units, activation=activation, name=f"decoder_dense_{index}")
        for index, units in enumerate(reversed(hidden_dims))
    ]
    output_layer = Dense(input_dim, activation=output_activation, name="reconstruction")

    input_tensor = Input(shape=(input_dim,), name="input")
    encoded = input_tensor
    for layer in encoder_layers:
        encoded = layer(encoded)
    latent = latent_layer(encoded)
    decoded = latent
    for layer in decoder_layers:
        decoded = layer(decoded)
    decoded = output_layer(decoded)

    autoencoder = Model(inputs=input_tensor, outputs=decoded, name="autoencoder")
    encoder = Model(inputs=input_tensor, outputs=latent, name="encoder")

    latent_input = Input(shape=(latent_dim,), name="latent_input")
    generated = latent_input
    for layer in decoder_layers:
        generated = layer(generated)
    generated = output_layer(generated)
    decoder = Model(inputs=latent_input, outputs=generated, name="decoder")
    return autoencoder, encoder, decoder


def split_frame_indices(
    frame_count: int,
    test_size: Optional[int] = None,
    test_fraction: float = 0.2,
    split_mode: str = "sequential",
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_count < 2:
        raise ValueError("At least two frames are required for training and testing")
    if test_size is None:
        test_size = max(1, int(round(frame_count * test_fraction)))
    if test_size <= 0 or test_size >= frame_count:
        raise ValueError("test_size must be between 1 and frame_count - 1")

    indices = np.arange(frame_count)
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        test_indices = np.sort(shuffled[:test_size])
        train_indices = np.sort(shuffled[test_size:])
        return train_indices, test_indices
    if split_mode != "sequential":
        raise ValueError("split_mode must be 'sequential' or 'random'")

    train_indices = indices[: frame_count - test_size]
    test_indices = indices[frame_count - test_size :]
    return train_indices, test_indices


def sample_latent_vectors(encoded_training_set: np.ndarray, sample_count: int, seed: int = DEFAULT_RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    mean = np.mean(encoded_training_set, axis=0)
    covariance = np.cov(encoded_training_set, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]])
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-6
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=mean, cov=covariance, size=sample_count)
    return samples, mean, covariance


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_matrix(path: Path, data: np.ndarray) -> None:
    np.savetxt(path, data, fmt="%.8f")


def _save_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_array(model, data: np.ndarray) -> np.ndarray:
    return np.asarray(model(data, training=False))


def train_coordinate_autoencoder(
    input_pdb: Union[str, Path],
    output_dir: Union[str, Path],
    latent_dim: int,
    atom_names: Optional[List[str]] = None,
    test_size: Optional[int] = None,
    test_fraction: float = 0.2,
    split_mode: str = "sequential",
    epochs: int = 100,
    batch_size: int = 40,
    optimizer: str = "adam",
    hidden_dims: tuple[int, ...] = (300, 50),
    sample_count: int = 100,
    seed: int = DEFAULT_RANDOM_SEED,
    label: str = "protein",
    verbose: int = 1,
) -> TrainingOutputs:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")

    set_random_seed(seed)
    atom_names = atom_names or list(DEFAULT_ATOM_NAMES)
    output_dir = _ensure_output_dir(output_dir)

    trajectory = PDBTrajectory.from_pdb(input_pdb)
    atom_indices = trajectory.select_indices(atom_names)
    selected_trajectory = trajectory.subset(atom_indices=atom_indices)
    selected_trajectory.subset(frame_indices=[0]).write_pdb(output_dir / "template_atoms.pdb")

    train_indices, test_indices = split_frame_indices(
        frame_count=selected_trajectory.n_frames,
        test_size=test_size,
        test_fraction=test_fraction,
        split_mode=split_mode,
        seed=seed,
    )

    flattened = selected_trajectory.flatten()
    x_train = flattened[train_indices]
    x_test = flattened[test_indices]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    autoencoder, encoder, decoder = build_dense_autoencoder(
        input_dim=x_train_scaled.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["mse"])
    history = autoencoder.fit(
        x_train_scaled,
        x_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_scaled, x_test_scaled),
        verbose=verbose,
    )

    encoded_train = _infer_array(encoder, x_train_scaled)
    encoded_test = _infer_array(encoder, x_test_scaled)
    decoded_test = _infer_array(autoencoder, x_test_scaled)
    decoded_test = scaler.inverse_transform(decoded_test)

    latent_samples, latent_mean, latent_cov = sample_latent_vectors(encoded_train, sample_count=sample_count, seed=seed)
    generated_flat = _infer_array(decoder, latent_samples)
    generated_flat = scaler.inverse_transform(generated_flat)

    encoder_path = output_dir / "encoder.keras"
    decoder_path = output_dir / "decoder.keras"
    scaler_path = output_dir / "scaler.pkl"
    metadata_path = output_dir / "metadata.json"

    encoder.save(encoder_path)
    decoder.save(decoder_path)
    joblib.dump(scaler, scaler_path)
    np.savez(output_dir / "latent_distribution.npz", mean=latent_mean, covariance=latent_cov)

    selected_trajectory.subset(frame_indices=train_indices).write_pdb(output_dir / f"{label}_training_set.pdb")
    selected_trajectory.subset(frame_indices=test_indices).write_pdb(output_dir / f"{label}_test_set.pdb")

    decoded_trajectory = PDBTrajectory(
        atom_records=selected_trajectory.atom_records,
        coordinates=decoded_test.reshape(decoded_test.shape[0], selected_trajectory.n_atoms, 3),
    )
    decoded_trajectory.write_pdb(output_dir / f"{label}_decoded_test_set.pdb")

    generated_trajectory = PDBTrajectory(
        atom_records=selected_trajectory.atom_records,
        coordinates=generated_flat.reshape(generated_flat.shape[0], selected_trajectory.n_atoms, 3),
    )
    generated_trajectory.write_pdb(output_dir / f"{label}_generated_samples.pdb")

    _write_matrix(output_dir / f"{label}_encoded_training_set.dat", encoded_train)
    _write_matrix(output_dir / f"{label}_encoded_test_set.dat", encoded_test)
    _write_matrix(output_dir / f"{label}_sampled_latent_vectors.dat", latent_samples)

    _save_metadata(
        metadata_path,
        {
            "label": label,
            "input_pdb": str(Path(input_pdb).resolve()),
            "latent_dim": latent_dim,
            "atom_names": atom_names,
            "train_frames": int(len(train_indices)),
            "test_frames": int(len(test_indices)),
            "split_mode": split_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
    )

    return TrainingOutputs(
        output_dir=output_dir,
        metadata_path=metadata_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        scaler_path=scaler_path,
    )


def generate_coordinate_structures(
    training_dir: Union[str, Path],
    sample_count: int,
    output_pdb: Optional[Union[str, Path]] = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    from tensorflow.keras.models import load_model

    training_dir = Path(training_dir)
    metadata = json.loads((training_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = joblib.load(training_dir / "scaler.pkl")
    decoder = load_model(training_dir / "decoder.keras")
    template = PDBTrajectory.from_pdb(training_dir / "template_atoms.pdb")
    latent_distribution = np.load(training_dir / "latent_distribution.npz")
    mean = latent_distribution["mean"]
    covariance = latent_distribution["covariance"]
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=mean, cov=covariance, size=sample_count)
    generated = _infer_array(decoder, samples)
    generated = scaler.inverse_transform(generated)
    generated_trajectory = PDBTrajectory(
        atom_records=template.atom_records,
        coordinates=generated.reshape(sample_count, template.n_atoms, 3),
    )
    output_path = Path(output_pdb) if output_pdb is not None else training_dir / f"{metadata['label']}_generated_from_saved_decoder.pdb"
    generated_trajectory.write_pdb(output_path)
    _write_matrix(training_dir / f"{metadata['label']}_generated_latent_vectors.dat", samples)
    return output_path


def _load_numeric_table(path: Union[str, Path], delimiter: Optional[str] = None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return np.asarray(data, dtype=float)


def train_tabular_autoencoder(
    input_table: Union[str, Path],
    output_dir: Union[str, Path],
    latent_dim: int,
    test_size: Optional[int] = None,
    test_fraction: float = 0.2,
    split_mode: str = "sequential",
    epochs: int = 100,
    batch_size: int = 40,
    optimizer: str = "adam",
    hidden_dims: tuple[int, ...] = (300, 50),
    sample_count: int = 100,
    seed: int = DEFAULT_RANDOM_SEED,
    label: str = "tabular",
    delimiter: Optional[str] = None,
    verbose: int = 1,
) -> TrainingOutputs:
    set_random_seed(seed)
    output_dir = _ensure_output_dir(output_dir)
    data = _load_numeric_table(input_table, delimiter=delimiter)
    train_indices, test_indices = split_frame_indices(
        frame_count=data.shape[0],
        test_size=test_size,
        test_fraction=test_fraction,
        split_mode=split_mode,
        seed=seed,
    )
    x_train = data[train_indices]
    x_test = data[test_indices]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    autoencoder, encoder, decoder = build_dense_autoencoder(
        input_dim=x_train.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["mse"])
    history = autoencoder.fit(
        x_train_scaled,
        x_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_scaled, x_test_scaled),
        verbose=verbose,
    )

    encoded_train = _infer_array(encoder, x_train_scaled)
    encoded_test = _infer_array(encoder, x_test_scaled)
    decoded_test = _infer_array(autoencoder, x_test_scaled)
    decoded_test = scaler.inverse_transform(decoded_test)

    latent_samples, latent_mean, latent_cov = sample_latent_vectors(encoded_train, sample_count=sample_count, seed=seed)
    generated = _infer_array(decoder, latent_samples)
    generated = scaler.inverse_transform(generated)

    encoder_path = output_dir / "encoder.keras"
    decoder_path = output_dir / "decoder.keras"
    scaler_path = output_dir / "scaler.pkl"
    metadata_path = output_dir / "metadata.json"

    encoder.save(encoder_path)
    decoder.save(decoder_path)
    joblib.dump(scaler, scaler_path)
    np.savez(output_dir / "latent_distribution.npz", mean=latent_mean, covariance=latent_cov)

    _write_matrix(output_dir / f"{label}_training_set.dat", x_train)
    _write_matrix(output_dir / f"{label}_test_set.dat", x_test)
    _write_matrix(output_dir / f"{label}_encoded_training_set.dat", encoded_train)
    _write_matrix(output_dir / f"{label}_encoded_test_set.dat", encoded_test)
    _write_matrix(output_dir / f"{label}_decoded_test_set.dat", decoded_test)
    _write_matrix(output_dir / f"{label}_generated_samples.dat", generated)
    _write_matrix(output_dir / f"{label}_sampled_latent_vectors.dat", latent_samples)

    _save_metadata(
        metadata_path,
        {
            "label": label,
            "input_table": str(Path(input_table).resolve()),
            "latent_dim": latent_dim,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "split_mode": split_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
    )

    return TrainingOutputs(
        output_dir=output_dir,
        metadata_path=metadata_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        scaler_path=scaler_path,
    )
