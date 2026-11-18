# PathSnap - GPS Trajectory Map Matching

## Project Overview

PathSnap is a machine learning-based GPS trajectory
map-matching system.

The system takes noisy GPS coordinates and predicts
the most likely roads travelled.

## Pipeline

GPS Data
-> Data Preprocessing
-> Road Candidate Generation
-> Feature Engineering
-> Machine Learning Models
-> HMM / Viterbi Decoder
-> Predicted Route
-> FastAPI
-> Evaluator Interface

## Team Responsibilities

### Person 1 - Data & Features
- Dataset exploration
- Data preprocessing
- Road candidate generation
- Feature engineering
- Logistic Regression baseline
- EDA visualizations

### Person 2 - Evaluation & Core Models
- Shared evaluation framework
- KNN
- Random Forest
- Model comparison

### Person 3 - Advanced Models & Tuning
- LightGBM / XGBoost
- MLP
- Hyperparameter tuning
- Evaluator UI

### Person 4 - HMM & Deployment
- HMM/Viterbi decoding
- FastAPI backend
- Model integration
- Deployment

## Repository Structure

- data/ : Raw and processed datasets
- src/data/ : Data preprocessing
- src/features/ : Feature engineering
- src/models/ : Machine learning models
- src/eval/ : Evaluation
- src/tuning/ : Hyperparameter tuning
- src/hmm/ : HMM decoder
- api/ : FastAPI backend
- ui/ : Evaluator interface
- notebooks/ : Experiments and EDA
- results/ : Evaluation results
- artifacts/ : Saved models
- reports/ : Project documentation

## Final results (development data only: train + student_test)

Candidate-level classifiers (labels are Person 1's nearest-route-arc pseudo-labels, 3-fold CV by trip on train; test = student_test):

| model | CV macro F1 | CV log loss | student_test macro F1 |
|---|---|---|---|
| Logistic regression (P1) | 0.710 | - | see `results/logistic_baseline_results.json` |
| KNN / Random Forest (P2) | see `results/core_model_comparison_table.csv` | | |
| LightGBM baseline / tuned (Optuna, 15 trials) | 0.853 / 0.849 | 0.170 / **0.149** | 0.869 / 0.867 |
| XGBoost | 0.853 | 0.161 | 0.870 |
| MLP | 0.848 | 0.176 | 0.867 |

Selected model: **tuned LightGBM** (lowest train-CV log loss), installed as `artifacts/model.joblib`.
Every model as an HMM emission: `results/model_comparison_hmm.csv` / `.png`.

Selected model + HMM (decoder tuned with Optuna on train only, 40 trials), student_test mean over 10 trips:

| | classifier only | classifier + HMM |
|---|---|---|
| route F1 (length-weighted, optimistic: ignores order) | 0.848 | **0.973** |
| pointwise accuracy vs pseudo-labels (stricter) | 0.823 | **0.897** |
| route discontinuities per trip | 206.9 | **1.4** |

Route F1 is nearly identical for all models once the HMM is applied (0.964-0.969), so pointwise accuracy is the
better model discriminator. Pointwise accuracy is agreement with a heuristic, not true ground truth.

## Run the whole project
    pip install -r requirements.txt -r api/requirements.txt lightgbm xgboost optuna
    D="<.../Student folder>"
    python -m src.features.build_candidates --data-root "$D" --split train          # + --split student_test
    python -m src.data.export_point_labels --split train                             # + student_test
    python -m src.visuals.eda_plots --data-root "$D"                                 # -> reports/figures/eda/
    python -m src.models.train_logistic ; python -m src.models.train_core_models     # P1, P2
    python -m src.models.train_advanced_models                                       # P3 (LightGBM/XGBoost/MLP + Optuna)
    python -m src.hmm.compare_models --data-dir "$D" --extra artifacts --point-labels-dir results/point_labels/student_test
    python -m src.hmm.tune_decoder --data-dir "$D" --scorer classifier --write-artifact
    PATHSNAP_DATA_DIR="$D" PATHSNAP_PARTITION=student_test PATHSNAP_SCORER=classifier uvicorn api.main:app --port 8000
    cd ui && python -m http.server 5173        # open http://localhost:5173
Details for the decoder/API are in `api/README.md`. `artifacts/models/random_forest` (320 MB) is not in git; regenerate with `train_core_models`.
