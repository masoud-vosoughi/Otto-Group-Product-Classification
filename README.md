# Otto Group Product Classification

An end-to-end multiclass classification project based on the [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-group-product-classification-challenge) on Kaggle.

## Project Overview

The objective is to predict product categories using 93 anonymized numerical features. The dataset contains nine target classes, and model performance is evaluated using **multiclass logarithmic loss**, where lower values indicate better probabilistic predictions.

The training dataset contains 61,878 labeled observations, while the test dataset contains 144,368 unlabeled observations.

## Methodology

The project follows a structured machine learning workflow:

1. **Data exploration:** Examined class distributions, missing values, feature sparsity, duplicate records, and train-test feature distributions.
2. **Baseline modeling:** Established a class-prior Dummy Classifier and a Logistic Regression baseline with feature standardization.
3. **Nonlinear modeling:** Trained LightGBM and compared boosting duration and tree complexity.
4. **Model evaluation:** Used a stratified 80/20 holdout split and three-fold stratified cross-validation.
5. **Final training:** Retrained the selected LightGBM configuration on the complete labeled dataset.
6. **Inference and submission:** Saved and reloaded the model, verified prediction consistency, and generated a Kaggle submission.

## Results

All reported values use multiclass log loss.

| Model / Evaluation | Log Loss |
|---|---:|
| Dummy Classifier — Holdout | 1.9505 |
| Logistic Regression — Holdout | 0.6311 |
| LightGBM — Holdout | 0.4687 |
| Logistic Regression — 3-Fold CV Mean | 0.6479 |
| LightGBM — 3-Fold CV Mean | 0.5091 |
| LightGBM — Kaggle Public | 0.46207 |
| LightGBM — Kaggle Private | **0.46128** |

LightGBM consistently outperformed the linear baseline. Although it exhibited a larger training-validation performance gap, its validation log loss was substantially lower.

The final configuration uses **15 leaves per tree** and **995 boosting iterations**, selected through holdout experiments with early stopping.

The Kaggle submission was evaluated after the competition deadline.

## Project Structure

- `notebooks/01_data_understanding.ipynb` — Data exploration, modeling experiments, evaluation, and final inference.
- `requirements.txt` — Direct Python dependencies and their tested versions.
- `data/raw/` — Kaggle datasets (not tracked by Git).
- `models/` — Saved model artifact (not tracked by Git).
- `submissions/` — Generated submission files (not tracked by Git).

## Setup and Reproduction

**Python:** 3.12

Create and activate a virtual environment, then install the dependencies listed in `requirements.txt`.

Download the competition files from Kaggle and place `train.csv`, `test.csv`, and `sampleSubmission.csv` inside `data/raw/`.

Open `notebooks/01_data_understanding.ipynb`, select the project's Python environment, and execute the notebook cells in order.

The notebook includes model training, validation, artifact persistence, inference verification, and submission generation.

## Limitations

- Features are anonymized, limiting domain-specific interpretation.
- Hyperparameters were selected using a single holdout split; subsequent cross-validation was used to examine performance stability rather than provide a fully independent estimate.
- Internal validation and Kaggle scores come from different evaluation settings and should not be treated as directly interchangeable.
- Trained model files and datasets are excluded from the repository and must be generated or obtained separately.