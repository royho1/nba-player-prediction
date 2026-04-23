"""
Classification Models for NBA 5th-Season Peak Prediction

Predicts whether a player's 5th season will be their highest-scoring
season out of their first five, using LDA, QDA, Random Forest, and XGBoost.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier


RANDOM_STATE = 15
TEST_SIZE = 0.3
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_and_prepare(filepath: str):
    """Load cleaned data and create classification target."""
    data = pd.read_csv(filepath)

    # Target: was the 5th season the player's best scoring season?
    data["season5_best"] = data["ptsseason5"] > data[
        ["ptsseason1", "ptsseason2", "ptsseason3", "ptsseason4"]
    ].max(axis=1)

    feature_cols = [
        "age", "draft_round", "draft_number",
        "player_height", "player_weight",
        "ptsseason1", "ptsseason2", "ptsseason3", "ptsseason4",
    ]

    X = data[feature_cols]
    y = data["season5_best"]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return data, X, y, train_X, test_X, train_y, test_y


def plot_confusion_matrix(cm, title, filename, cmap="Blues"):
    """Save a confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()


def run_lda_qda(train_X, test_X, train_y, test_y):
    """Fit LDA and QDA classifiers."""
    results = {}
    roc_data = {}

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_y)
    y_pred = lda.predict(test_X)
    y_proba = lda.predict_proba(test_X)[:, 1]

    acc = accuracy_score(test_y, y_pred)
    results["LDA"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["LDA"] = (fpr, tpr)

    print(f"LDA Accuracy: {acc:.4f}")
    print(classification_report(test_y, y_pred))

    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix - LDA", "cm_lda.png", "Greens")

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_X, train_y)
    y_pred = qda.predict(test_X)
    y_proba = qda.predict_proba(test_X)[:, 1]

    acc = accuracy_score(test_y, y_pred)
    results["QDA"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["QDA"] = (fpr, tpr)

    print(f"QDA Accuracy: {acc:.4f}")
    print(classification_report(test_y, y_pred))

    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix - QDA", "cm_qda.png", "Blues")

    return results, roc_data


def run_random_forest(train_X, test_X, train_y, test_y):
    """Fit Random Forest with and without hyperparameter tuning."""
    results = {}
    roc_data = {}

    # Baseline RF
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(train_X, train_y)
    y_proba = rf.predict_proba(test_X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(test_y, y_pred)
    results["Random Forest"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["Random Forest"] = (fpr, tpr)

    print(f"Random Forest Accuracy: {acc:.4f}")
    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix - Random Forest", "cm_rf.png", "Purples")

    # Tuned RF via RandomizedSearchCV
    param_grid = {
        "n_estimators": [int(x) for x in np.linspace(100, 1000, 10)],
        "max_depth": [int(x) for x in np.linspace(10, 100, 10)] + [None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_grid,
        n_iter=100,
        cv=3,
        verbose=0,
        random_state=12,
    )
    rf_search.fit(train_X, train_y)

    print(f"Best RF params: {rf_search.best_params_}")

    y_proba = rf_search.predict_proba(test_X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(test_y, y_pred)
    results["Random Forest (Tuned)"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["Random Forest (Tuned)"] = (fpr, tpr)

    print(f"Random Forest (Tuned) Accuracy: {acc:.4f}")
    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(
        cm, "Confusion Matrix - RF Tuned", "cm_rf_tuned.png", "Reds"
    )

    return results, roc_data


def run_feature_engineering(X, y, train_X, test_X, train_y, test_y):
    """Add engineered features and re-run Random Forest."""
    results = {}
    roc_data = {}

    X_eng = X.copy()
    X_eng["height_x_weight"] = X_eng["player_height"] * X_eng["player_weight"]
    X_eng["improving_player"] = (
        (X_eng["ptsseason4"] > X_eng["ptsseason3"])
        & (X_eng["ptsseason3"] > X_eng["ptsseason2"])
    )

    train_X_e, test_X_e, train_y_e, test_y_e = train_test_split(
        X_eng, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        min_samples_split=5,
        max_depth=90,
        bootstrap=True,
        class_weight="balanced",
        random_state=12,
    )
    rf.fit(train_X_e, train_y_e)
    y_proba = rf.predict_proba(test_X_e)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(test_y_e, y_pred)
    results["RF w/ Feature Engineering"] = acc
    fpr, tpr, _ = roc_curve(test_y_e, y_proba)
    roc_data["RF w/ Feature Engineering"] = (fpr, tpr)

    print(f"RF w/ Feature Engineering Accuracy: {acc:.4f}")

    return results, roc_data


def run_xgboost(train_X, test_X, train_y, test_y):
    """Fit XGBoost with class imbalance handling."""
    results = {}
    roc_data = {}

    ratio = train_y.value_counts()[False] / train_y.value_counts()[True]

    # Baseline XGBoost
    xgb = XGBClassifier(
        scale_pos_weight=ratio,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(train_X, train_y)
    y_proba = xgb.predict_proba(test_X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(test_y, y_pred)
    results["XGBoost"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["XGBoost"] = (fpr, tpr)

    print(f"XGBoost Accuracy: {acc:.4f}")
    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm, "Confusion Matrix - XGBoost", "cm_xgb.png", "Reds")

    # Tuned XGBoost
    grid_xgb = {
        "n_estimators": [int(x) for x in np.linspace(100, 1000, 10)],
        "max_depth": [int(x) for x in np.linspace(10, 100, 10)] + [None],
        "learning_rate": [0.01, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
        "colsample_bytree": [0.8, 1],
        "gamma": [0, 0.1, 0.3],
        "scale_pos_weight": [ratio],
    }

    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        param_distributions=grid_xgb,
        n_iter=100,
        cv=3,
        verbose=0,
        random_state=12,
    )
    xgb_search.fit(train_X, train_y)

    print(f"Best XGB params: {xgb_search.best_params_}")

    y_proba = xgb_search.predict_proba(test_X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    acc = accuracy_score(test_y, y_pred)
    results["XGBoost (Tuned)"] = acc
    fpr, tpr, _ = roc_curve(test_y, y_proba)
    roc_data["XGBoost (Tuned)"] = (fpr, tpr)

    print(f"XGBoost (Tuned) Accuracy: {acc:.4f}")
    cm = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(
        cm, "Confusion Matrix - XGBoost Tuned", "cm_xgb_tuned.png", "Blues"
    )

    return results, roc_data


def plot_all_roc(all_roc_data):
    """Plot ROC curves for all classification models."""
    plt.figure(figsize=(12, 8))
    for name, (fpr, tpr) in all_roc_data.items():
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Classification Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()


def plot_class_distribution(data):
    """Plot the class distribution of the target variable."""
    sns.countplot(x=data["season5_best"])
    plt.title("Did Player Perform Best in 5th Season?")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train5final.csv")
    data, X, y, train_X, test_X, train_y, test_y = load_and_prepare(data_path)

    plot_class_distribution(data)

    all_results = {}
    all_roc = {}

    print("=" * 60)
    print("LDA / QDA")
    print("=" * 60)
    res, roc = run_lda_qda(train_X, test_X, train_y, test_y)
    all_results.update(res)
    all_roc.update(roc)

    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    res, roc = run_random_forest(train_X, test_X, train_y, test_y)
    all_results.update(res)
    all_roc.update(roc)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    res, roc = run_feature_engineering(X, y, train_X, test_X, train_y, test_y)
    all_results.update(res)
    all_roc.update(roc)

    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)
    res, roc = run_xgboost(train_X, test_X, train_y, test_y)
    all_results.update(res)
    all_roc.update(roc)

    # Summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)
    for name, acc in all_results.items():
        print(f"  {name:<35} Accuracy: {acc:.4f}")

    plot_all_roc(all_roc)
    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
