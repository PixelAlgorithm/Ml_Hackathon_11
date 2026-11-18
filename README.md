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
