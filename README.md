# Generative IDPs

This repository now contains a functional Python package for training, sampling, and analyzing generative autoencoders for intrinsically disordered proteins. The original research scripts are still present under the same folders, but they have been converted into thin wrappers around a maintained codebase in `src/generative_idps`.

## What Changed

- Removed the hard dependency on the unpublished `biobox` module by adding a native multi-model PDB parser and writer.
- Replaced hardcoded paths, frame ranges, and atom selections with explicit CLI arguments.
- Added packaging metadata so the repo can be installed and run consistently.
- Centralized the coordinate, dihedral, distance, clustering, histogram, and RMSD workflows behind one CLI.
- Added tests for the new PDB parsing and RMSD logic.

## Repository Layout

```text
generative_IDPs/
├── src/generative_idps/
│   ├── cli.py                # Unified command-line interface
│   ├── training.py           # Coordinate and tabular autoencoder workflows
│   ├── trajectory.py         # Multi-model PDB parsing and writing
│   ├── analysis.py           # Clustering, plotting, and RMSD utilities
│   └── constants.py          # Shared defaults
├── Implementation_codes/     # Backward-compatible wrappers and sample PDB
├── Plot_histogram/           # Backward-compatible wrapper
├── RMSD/                     # Backward-compatible wrappers
├── tests/                    # Lightweight regression tests
├── pyproject.toml            # Installable package metadata
└── requirements.txt          # Direct pip install dependencies
```

## Installation

### 1. Create and activate a virtual environment

```bash
cd /Users/aayush/Desktop/Repo/generative_IDPs
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 3. Optional developer dependencies

```bash
python -m pip install -e .[dev]
```

## Quick Start

The repository ships with a sample coordinate trajectory at `Implementation_codes/polyq_1000frames.pdb`.
It also now includes a small numeric matrix for the tabular workflows at `examples/sample_tabular_data.dat`.

### Train a coordinate autoencoder

```bash
generative-idps train-coordinate \
    --input-pdb Implementation_codes/polyq_1000frames.pdb \
    --output-dir outputs/polyq_demo \
    --latent-dim 13 \
    --test-size 200 \
    --epochs 25 \
    --batch-size 40 \
    --sample-count 100 \
    --label polyq
```

This writes:

- Encoded train and test sets
- Saved encoder and decoder models in Keras format
- A saved scaler and latent distribution
- Reconstructed test structures
- Newly generated PDB structures sampled from the latent space

### Generate new structures from a saved model

```bash
generative-idps generate-coordinate \
    --training-dir outputs/polyq_demo \
    --sample-count 50 \
    --output-pdb outputs/polyq_demo/polyq_resampled.pdb
```

### Train on tabular dihedral or distance data

```bash
generative-idps train-tabular \
    --input-table examples/sample_tabular_data.dat \
    --output-dir outputs/dihedral_demo \
    --latent-dim 3 \
    --test-fraction 0.2 \
    --epochs 10 \
    --batch-size 8 \
    --label dihedral
```

### Cluster an encoded dataset

```bash
generative-idps cluster \
    --input outputs/polyq_demo/polyq_encoded_training_set.dat \
    --output-dir outputs/polyq_demo/clustering \
    --clusters 8
```

To cluster only selected latent dimensions:

```bash
generative-idps cluster \
    --input outputs/polyq_demo/polyq_encoded_training_set.dat \
    --output-dir outputs/polyq_demo/clustering_subset \
    --clusters 8 \
    --columns 0,1,2,3
```

### Plot training, test, and generated latent histograms

```bash
generative-idps plot-histogram \
    --training outputs/polyq_demo/polyq_encoded_training_set.dat \
    --test outputs/polyq_demo/polyq_encoded_test_set.dat \
    --generated outputs/polyq_demo/polyq_sampled_latent_vectors.dat \
    --output outputs/polyq_demo/latent_histogram.png \
    --x-dim 0 \
    --y-dim 1
```

### Compare trajectories with RMSD

```bash
generative-idps rmsd \
    --reference-pdb outputs/polyq_demo/polyq_test_set.pdb \
    --candidate-pdb outputs/polyq_demo/polyq_generated_samples.pdb \
    --output outputs/polyq_demo/min_rmsd.txt
```

Use `--unique-match` only for small datasets, because it constructs the full pairwise RMSD matrix before optimal assignment.

## Legacy Entry Points

The original script names still work, but they now delegate to the unified CLI.

- `Implementation_codes/autoencoder_train_generate.py`: accepts the original key-value input file format.
- `Implementation_codes/generate_new_IDPs.py`: forwards to `generate-coordinate`.
- `Implementation_codes/dihedral_train_generate.py`: forwards to `train-tabular --label dihedral`.
- `Implementation_codes/distance_train_generate.py`: forwards to `train-tabular --label distance`.
- `Implementation_codes/clustering.py`: forwards to `cluster`.
- `Plot_histogram/hist.py`: forwards to `plot-histogram`.
- `RMSD/rmsd_one2many.py`: forwards to `rmsd`.
- `RMSD/mini_rmsd_one2many.py`: forwards to `extract-min-rmsd`.

## Example Legacy Workflow

You can still use the original key-value input style:

```text
dimensions 13
batch_size 40
out_label polyq
trajectory ./Implementation_codes/polyq_1000frames.pdb
test_size 200
epochs 25
out_folder ./outputs/legacy_polyq
optimizer adam
```

Run it with:

```bash
python Implementation_codes/autoencoder_train_generate.py path/to/input.txt
```

## Validation

Run the tests with:

```bash
pytest
```

You can also run the sample shell wrapper:

```bash
bash Implementation_codes/run_python.sh
```

Validated locally in this repository:

- Coordinate training on `Implementation_codes/polyq_1000frames.pdb`
- Coordinate generation from the saved decoder
- Clustering of the encoded latent space
- Histogram generation
- RMSD comparison
- Tabular training on `examples/sample_tabular_data.dat`
- Legacy wrapper invocation through `Implementation_codes/autoencoder_train_generate.py`

## Notes

- The coordinate pipeline assumes all models in a PDB trajectory share the same atom order.
- TensorFlow is now used through modern `tensorflow.keras` APIs instead of the deprecated standalone Keras calls from the original scripts.
- On macOS, dependency markers install `tensorflow-macos` and pin `urllib3<2` to avoid the LibreSSL warning emitted by Apple-provided Python builds.
- `RMSD/rmsdtrj.c` remains an archival prototype. The maintained RMSD workflow is the Python CLI because the C file depends on missing research-only headers from the original environment.






