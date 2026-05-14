# Cross-Period Spatial Correlation Project

This repository contains the workflow I used for PCA-based cross-period spatial correlation modeling of ground-motion residuals.

The main idea is to first reduce the period-dependent within-event residuals into principal component scores, and then fit spatial correlation models to each normalized principal component. The workflow is currently notebook-based, with the main analysis in `pipeline.ipynb` and the NumPyro model definitions in `models_gpu.py`.

## What this repository does

The workflow has three main steps.

First, it reads nine period-dependent `scaled_deltaW` columns from `data/data.csv` and performs PCA across periods. The PCA results are saved so that the loading matrix, explained variance ratios, and normalized PC scores can be checked separately.

Second, the data are grouped by earthquake event. For each event, the notebook precomputes the matrices needed for spatial correlation inference, including inter-station distance, azimuthal separation, and site dissimilarity based on `Vs30`.

Third, the notebook fits two spatial correlation models to each normalized principal component:

- `modelE`: a distance-only spatial correlation model;
- `modelEAS`: an extended model that includes distance, azimuth/path, and site terms.

Posterior samples from the inference runs are saved in `inference_results/`.

## Repository structure

```text
.
|-- data/
|   `-- data.csv
|-- pca_results/
|-- inference_results/
|-- models_gpu.py
|-- pipeline.ipynb
`-- README.md
