# Multi-Period PCA and EAS Spatial Correlation Inference

This repository contains a notebook-based workflow for:

- computing PCA from nine period-dependent `scaled_deltaW` features in `data/data.csv`;
- exporting PCA loadings, explained variance ratios, and normalized principal component scores; and
- performing Bayesian spatial-correlation inference for `normalized_PC1` through `normalized_PC9`.

The main workflow is implemented in `pipeline.ipynb`, and the model definitions are provided in `models_gpu.py`.

## Project Structure

```text
.
|-- data/
|   `-- data.csv
|-- pca_results/
|-- inference_results/
|-- models_gpu.py
|-- pipeline.ipynb
`-- README.md
```

## Overview

### 1. PCA analysis

The notebook reads the following nine input columns from `data/data.csv`:

- `scaled_deltaWT001`
- `scaled_deltaWT003`
- `scaled_deltaWT006`
- `scaled_deltaWT010`
- `scaled_deltaWT030`
- `scaled_deltaWT060`
- `scaled_deltaWT100`
- `scaled_deltaWT300`
- `scaled_deltaWT600`

It then fits a 9-component PCA model and writes the results to `pca_results/`:

- `pca_loadings.csv`: loadings of each period on each principal component
- `pca_variance_ratio.csv`: explained variance ratio of each principal component
- `data_with_normalized_pc_scores.csv`: original data with `normalized_PC1` through `normalized_PC9` appended

### 2. Event-level preprocessing

The data are grouped by `eqid`, and for each earthquake event the notebook precomputes:

- `distE`: inter-station epicentral distance matrix
- `angdeg`: azimuthal separation matrix in degrees
- `soil`: site dissimilarity matrix based on `vs30`

These matrices are reused across all principal components and both models to avoid redundant computation during inference.

### 3. Bayesian spatial correlation inference

`models_gpu.py` defines two NumPyro models:

- `modelE`: a spatial correlation model using only the distance term
- `modelEAS`: a spatial correlation model combining distance, azimuth/path, and site terms

The notebook runs both models for `normalized_PC1` through `normalized_PC9` and saves the posterior samples to `inference_results/`.

After a full run, the expected output files include:

- `modelE_normalized_PC1_posterior.csv` through `modelE_normalized_PC9_posterior.csv`
- `modelEAS_normalized_PC1_posterior.csv` through `modelEAS_normalized_PC9_posterior.csv`

## Input Data Requirements

`data/data.csv` is expected to contain at least the following fields:

- event and metadata: `eqid`, `region`, `magnitude`
- geometry and site information: `epi_dist`, `epi_azimuth`, `vs30`
- PCA inputs: the nine `scaled_deltaW...` columns listed above

`epi_azimuth` is expected to be in radians, because the angular-distance calculations in `models_gpu.py` use trigonometric functions directly.

## Environment

Recommended Python packages:

```bash
pip install pandas scikit-learn numpy jax numpyro
```

If GPU acceleration is needed, install the JAX build that matches your CUDA environment.

## Usage

1. Place the input dataset at `data/data.csv`.
2. Open `pipeline.ipynb`.
3. Run all notebook cells in order.
4. Check `pca_results/` for PCA outputs.
5. Check `inference_results/` for posterior sample CSV files.

## Model Notes

`models_gpu.py` enables `jax_enable_x64=True` by default for numerical stability. The notebook also detects whether JAX is running on GPU or CPU and configures NumPyro accordingly.

The current MCMC defaults in the notebook are:

- `num_chains = 1`
- `num_warmup = 500`
- `num_samples = 500`
- `base_seed = 31`

If you need more stable posterior estimates, you can increase the warmup and sample counts.

## Outputs

### PCA outputs

- `pca_loadings.csv`: PCA loading matrix
- `pca_variance_ratio.csv`: explained variance ratio for each principal component
- `data_with_normalized_pc_scores.csv`: full dataset with normalized principal component scores added

### Inference outputs

Each posterior CSV corresponds to one model and one principal component, and contains the MCMC samples for that combination.

- `modelE` parameters: `LE`, `gammaE`
- `modelEAS` parameters: `LE`, `gammaE`, `LA`, `LS`, `w`

## Notes

- The repository currently uses `pipeline.ipynb` as the main entry point rather than a standalone CLI script.
- `inference_results/` may be empty until the inference section of the notebook has been executed.
- The workflow is designed to perform PCA first and then model each normalized principal component separately, allowing the spatial correlation structure of multi-period residual behavior to be analyzed component by component.
