"""
train.py
========
Heart Disease Classification — CDC BRFSS 2020
Hochschule Pforzheim · BIS3012 · 2026

Trains three classifiers (Logistic Regression, Random Forest, SVM) using
GridSearchCV with 5-fold stratified cross-validation. All models use
class_weight='balanced' to handle the real-world class imbalance (8.6% disease).

Usage:
    python src/train.py

Requires:
    data/X_train.csv, data/y_train.csv  (run preprocess.py first)

Output:
    Prints best parameters and cross-validation scores for each model.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR     = "data"
RANDOM_STATE = 42
CV_FOLDS     = 5
SCORING      = "f1"   # F1 balances precision and recall for imbalanced data

# ── Hyperparameter grids ──────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'   # compensates for 8.6% disease rate
        ),
        "grid": {"C": [0.01, 0.1, 0.5, 1, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        "grid": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Support Vector Machine": {
        "model": SVC(
            kernel='rbf',
            probability=True,          # required for predict_proba / ROC curves
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        "grid": {
            "C":     [0.1, 1, 10, 100],
            "gamma": ['scale', 'auto', 0.01, 0.001]
        }
    }
}


def load_training_data(data_dir: str):
    """Load preprocessed training data."""
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    print(f"Training data loaded: {X_train.shape}")
    print(f"  Disease rate: {y_train.mean()*100:.1f}%")
    return X_train, y_train


def scale_features(X_train: pd.DataFrame):
    """
    Fit StandardScaler on training data only.

    IMPORTANT: The scaler is fitted ONLY on training data.
    Fitting on all data would cause data leakage — test set information
    would influence the training process, producing falsely optimistic results.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"Features scaled (mean≈0, std≈1). Scaler fitted on training data only.")
    return X_train_scaled, scaler


def train_all_models(X_train_scaled, y_train) -> dict:
    """Run GridSearchCV for all three models and return best estimators."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n{'='*60}")
    print(f"GridSearchCV — {CV_FOLDS}-fold stratified CV — scoring: {SCORING}")
    print(f"class_weight='balanced' applied to all models")
    print(f"{'='*60}")

    for name, config in PARAM_GRIDS.items():
        print(f"\n[{name}]")
        print(f"  Parameter grid: {config['grid']}")

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["grid"],
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,       # use all CPU cores
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        best = grid_search.best_estimator_

        # Cross-validation accuracy with best params
        cv_scores = cross_val_score(best, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        print(f"  Best params:  {grid_search.best_params_}")
        print(f"  CV accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model":      best,
            "best_params": grid_search.best_params_,
            "cv_mean":    cv_scores.mean(),
            "cv_std":     cv_scores.std()
        }

    return results


def main():
    print("=" * 55)
    print("TRAINING — Heart Disease Classification")
    print("=" * 55)

    print("\n[1/3] Loading training data...")
    X_train, y_train = load_training_data(DATA_DIR)

    print("\n[2/3] Scaling features...")
    X_train_scaled, scaler = scale_features(X_train)

    print("\n[3/3] Training models with GridSearchCV...")
    results = train_all_models(X_train_scaled, y_train)

    print("\n" + "="*55)
    print("TRAINING COMPLETE — Summary")
    print("="*55)
    print(f"{'Model':<30} {'CV Acc':>8} {'±Std':>7}")
    print("-"*50)
    for name, info in results.items():
        print(f"{name:<30} {info['cv_mean']:>8.4f} {info['cv_std']:>7.4f}")

    print("\nRun 'python src/evaluate.py' next.")
    return results, scaler


if __name__ == "__main__":
    main()
