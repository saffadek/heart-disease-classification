"""
evaluate.py
===========
Heart Disease Classification — CDC BRFSS 2020
Hochschule Pforzheim · BIS3012 · 2026

Evaluates all three trained models on the held-out test set.
Generates confusion matrices, ROC curves, feature importance chart,
and prints a full classification report for each model.

Usage:
    python src/evaluate.py

Requires:
    data/X_train.csv, data/X_test.csv,
    data/y_train.csv, data/y_test.csv
    (run preprocess.py then train.py first)

Output:
    Console: full classification reports + metric summary table
    Figures: confusion_matrices.png, roc_curves.png, feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, recall_score, precision_score, f1_score,
    ConfusionMatrixDisplay
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR     = "data"
RANDOM_STATE = 42
CV_FOLDS     = 5

# Verified best parameters from GridSearchCV
BEST_PARAMS = {
    "Logistic Regression": {
        "model": LogisticRegression(
            C=0.01, max_iter=1000,
            random_state=RANDOM_STATE, class_weight='balanced'
        )
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            max_depth=5, min_samples_split=10, n_estimators=100,
            random_state=RANDOM_STATE, class_weight='balanced'
        )
    },
    "Support Vector Machine": {
        "model": SVC(
            C=0.1, gamma=0.001, kernel='rbf',
            probability=True, random_state=RANDOM_STATE, class_weight='balanced'
        )
    }
}


def load_data(data_dir: str):
    """Load preprocessed train and test sets."""
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    print(f"Data loaded: {len(X_train)} train, {len(X_test)} test patients")
    print(f"Test disease rate: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    """Scale features — fit on train only, transform both."""
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    return X_tr, X_te, scaler


def evaluate_models(X_tr, X_te, y_train, y_test, X_train) -> dict:
    """Train with best params and evaluate all three models."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n{'='*70}")
    print(f"EVALUATION — All Models on {len(y_test)}-patient Held-Out Test Set")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'CV Acc':>8} {'±Std':>6} {'Test Acc':>10} {'Recall':>8} {'AUC':>7}")
    print("-"*70)

    for name, config in BEST_PARAMS.items():
        model = config["model"]
        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        cm          = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "test_acc": accuracy_score(y_test, y_pred),
            "recall":   recall_score(y_test, y_pred),
            "precision":precision_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "auc":      roc_auc,
            "fpr":      fpr, "tpr": tpr,
            "cm":       cm,
            "tn":tn, "fp":fp, "fn":fn, "tp":tp
        }

        print(f"{name:<30} {cv_scores.mean():>8.4f} {cv_scores.std():>6.4f} "
              f"{accuracy_score(y_test,y_pred):>10.4f} "
              f"{recall_score(y_test,y_pred):>8.4f} {roc_auc:>7.4f}")

    return results


def print_classification_reports(results: dict, y_test):
    """Print full classification report for each model."""
    for name, info in results.items():
        print(f"\n{'='*55}")
        print(f" {name}")
        print(f"{'='*55}")
        print(f" Best params: {BEST_PARAMS[name]['model'].get_params()}")
        print(f" TN={info['tn']} FP={info['fp']} FN={info['fn']} TP={info['tp']}")
        print()
        print(classification_report(
            y_test, info['y_pred'],
            target_names=['No Disease', 'Heart Disease'],
            digits=3
        ))


def plot_confusion_matrices(results: dict, y_test):
    """Plot confusion matrices for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, (name, info) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(
            info['cm'],
            display_labels=['No Disease', 'Heart Disease']
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        tn,fp,fn,tp = info['tn'],info['fp'],info['fn'],info['tp']
        ax.set_title(
            f"{name}\n"
            f"Acc: {info['test_acc']:.1%}  "
            f"Recall: {info['recall']:.1%}  "
            f"FN: {fn}",
            fontsize=10
        )

    plt.suptitle(
        f"Confusion Matrices — CDC BRFSS 2020 — {len(y_test)} Test Patients\n"
        f"class_weight='balanced' · GridSearchCV · F1 scoring",
        fontsize=12, fontweight='bold', y=1.04
    )
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
    print("Saved: confusion_matrices.png")
    plt.show()


def plot_roc_curves(results: dict):
    """Plot ROC curves for all three models."""
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for (name, info), color in zip(results.items(), colors):
        plt.plot(
            info['fpr'], info['tpr'],
            color=color, lw=2.5,
            label=f"{name}  (AUC = {info['auc']:.3f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Baseline (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves — Heart Disease Classification\nCDC BRFSS 2020 · 2,059 patients',
              fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
    print("Saved: roc_curves.png")
    plt.show()


def plot_feature_importance(results: dict, feature_names):
    """Plot feature importance from the Random Forest model."""
    rf = results["Random Forest"]["model"]
    fi = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)

    # Highlight top 5 in red
    colors = ['#e74c3c' if i >= len(fi) - 5 else '#3498db' for i in range(len(fi))]

    plt.figure(figsize=(9, 7))
    plt.barh(fi['Feature'], fi['Importance'], color=colors, edgecolor='white', linewidth=0.5)
    plt.xlabel('Importance Score (Gini Mean Decrease in Impurity)')
    plt.title(
        'Random Forest — Feature Importance\n'
        'Red = top 5 most influential features',
        fontweight='bold'
    )
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    print("Saved: feature_importance.png")
    plt.show()

    print("\nTop 10 features ranked:")
    for i, (_, row) in enumerate(fi.iloc[::-1].head(10).iterrows(), 1):
        print(f"  {i:2}. {row['Feature']:<22} {row['Importance']*100:.1f}%")


def print_summary(results: dict):
    """Print final summary table."""
    print("\n" + "="*70)
    print("FINAL SUMMARY — Best Model Selection")
    print("="*70)
    print(f"{'Model':<30} {'Test Acc':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-"*70)

    best_name = max(results, key=lambda m: results[m]['auc'])
    for name, info in results.items():
        tag = ' ← BEST (AUC)' if name == best_name else ''
        print(f"{name:<30} {info['test_acc']:>10.4f} {info['recall']:>8.4f} "
              f"{info['f1']:>8.4f} {info['auc']:>8.4f}{tag}")

    print("="*70)
    b = results[best_name]
    print(f"\nBest model (by AUC):  {best_name}")
    print(f"Test accuracy:         {b['test_acc']:.1%}")
    print(f"Disease recall:        {b['recall']:.1%}")
    print(f"AUC-ROC:               {b['auc']:.3f}")
    print(f"\nNote: SVM achieves highest raw accuracy ({results['Support Vector Machine']['test_acc']:.1%})")
    print(f"but lowest disease recall ({results['Support Vector Machine']['recall']:.1%}) —")
    print(f"it exploits the 91.4% majority class. AUC is the more reliable metric.")


def main():
    print("=" * 55)
    print("EVALUATION — Heart Disease Classification")
    print("=" * 55)

    print("\n[1/4] Loading data...")
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)

    print("\n[2/4] Scaling features...")
    X_tr, X_te, scaler = scale(X_train, X_test)

    print("\n[3/4] Training and evaluating models...")
    results = evaluate_models(X_tr, X_te, y_train, y_test, X_train)

    print("\n[4/4] Generating reports and charts...")
    print_classification_reports(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results)
    plot_feature_importance(results, X_train.columns)
    print_summary(results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
