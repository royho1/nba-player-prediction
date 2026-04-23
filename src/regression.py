"""
Regression Models for NBA 5th-Season Points Prediction

Compares OLS, Decision Tree, Random Forest, and Gradient Boosted Trees
with hyperparameter tuning and cross-validation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


FEATURES = [
    "age",
    "draft_number",
    "player_height",
    "player_weight",
    "ptsseason1",
    "ptsseason2",
    "ptsseason3",
    "ptsseason4",
]
TARGET = "ptsseason5"
RANDOM_STATE = 15
TEST_SIZE = 0.3
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_data(filepath: str):
    """Load cleaned data and return train/test splits."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return df, X_train, X_test, y_train, y_test


def run_ols(X_train, X_test, y_train, y_test):
    """Fit OLS model and print results."""
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train_c).fit()
    print(model.summary())

    y_pred = model.predict(X_test_c)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nOLS Test RMSE: {rmse:.4f}")
    print(f"OLS Test R²:   {r2:.4f}")

    # Q-Q plot
    residuals = y_test - y_pred
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of OLS Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "qq_plot_residuals.png"), dpi=150)
    plt.close()

    # Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Points in Season 5")
    plt.ylabel("Predicted Points in Season 5")
    plt.title("OLS: Actual vs. Predicted Points")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    return rmse, r2


def run_tree_models(X_train, X_test, y_train, y_test):
    """Fit Decision Tree, Random Forest, and Gradient Boosted Tree models."""
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), [])],
        remainder="passthrough",
    )

    results = {}

    # --- Simple Decision Tree ---
    tree_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ])
    tree_pipe.fit(X_train, y_train)
    y_pred = tree_pipe.predict(X_test)
    results["Simple Decision Tree"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    # Tree visualization
    fitted_tree = tree_pipe.named_steps["regressor"]
    feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=10,
    )
    plt.title("Decision Tree (limited to depth=3)")
    plt.savefig(
        os.path.join(FIGURES_DIR, "decision_tree_diagram.png"), dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --- Random Forest ---
    rf_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    results["Random Forest"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
    }

    # Feature importance plot
    feature_names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
    importances = rf_pipe.named_steps["regressor"].feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importances.png"), dpi=150)
    plt.close()

    # Residual plot
    residuals = y_test - y_pred_rf
    sns.scatterplot(x=y_pred_rf, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Points (Season 5)")
    plt.ylabel("Residuals")
    plt.title("Random Forest Residual Plot")
    plt.savefig(os.path.join(FIGURES_DIR, "residual_plot.png"), dpi=150)
    plt.close()

    # --- Tuned Random Forest ---
    param_grid_rf = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [None, 5, 10],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
        "regressor__max_features": ["sqrt"],
    }
    grid_rf = GridSearchCV(
        Pipeline([
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]),
        param_grid_rf,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    y_pred = grid_rf.best_estimator_.predict(X_test)
    results["Random Forest (Tuned)"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print(f"Best RF params: {grid_rf.best_params_}")

    # --- Gradient Boosted Trees ---
    gb_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
        )),
    ])
    gb_pipe.fit(X_train, y_train)
    y_pred = gb_pipe.predict(X_test)
    results["Gradient Boosted Trees"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    # --- Tuned Gradient Boosted Trees ---
    param_grid_gb = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__max_depth": [3, 5, 7],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
        "regressor__subsample": [0.8, 1.0],
    }
    grid_gb = GridSearchCV(
        Pipeline([
            ("preprocess", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]),
        param_grid_gb,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_gb.fit(X_train, y_train)
    y_pred = grid_gb.best_estimator_.predict(X_test)
    results["Gradient Boosted Trees (Tuned)"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print(f"Best GB params: {grid_gb.best_params_}")

    return results


def run_cross_validation(df):
    """Run 3-fold cross-validation on regression models."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosted Trees": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    names, rmses = [], []
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
        )
        avg_rmse = np.sqrt(-scores.mean())
        names.append(name)
        rmses.append(avg_rmse)
        print(f"{name} - 3-Fold CV Avg RMSE: {avg_rmse:.4f}")

    plt.bar(names, rmses)
    plt.ylabel("Average RMSE on Training Set")
    plt.title("3-Fold Cross-Validation RMSE Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cv_rmse_comparison.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train5final.csv")
    df, X_train, X_test, y_train, y_test = load_data(data_path)

    print("=" * 60)
    print("ORDINARY LEAST SQUARES")
    print("=" * 60)
    ols_rmse, ols_r2 = run_ols(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("TREE-BASED MODELS")
    print("=" * 60)
    tree_results = run_tree_models(X_train, X_test, y_train, y_test)

    print("\nRegression Results Summary:")
    print(f"  {'Model':<35} {'RMSE':>8} {'R²':>8}")
    print(f"  {'OLS':<35} {ols_rmse:>8.4f} {ols_r2:>8.4f}")
    for name, metrics in tree_results.items():
        print(f"  {name:<35} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    run_cross_validation(df)


if __name__ == "__main__":
    main()
