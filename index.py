"""
    Investigating
"""

import json

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data.clean_data import fetch_check


def load_and_preprocess_data():
    """
    Creates dataframe using fetch_check(), then creates a new HeartDisease column which will
    be used as the target column, then prepares the x, y datasets
    """

    df = fetch_check()
    df["HeartDisease"] = ((df["HadHeartAttack"] == 1) | (df["HadAngina"] == 1)).astype(
        int
    )

    print("\nHeartDisease column distribution:")
    print(df["HeartDisease"].value_counts(normalize=True))

    x = df.drop(["HeartDisease"], axis=1)
    y = df["HeartDisease"]

    for column in x.columns:
        if x[column].dtype in ["int64", "float64"]:
            x[column].fillna(x[column].median(), inplace=True)
        else:
            x[column].fillna(x[column].mode()[0], inplace=True)

    return x, y


def split_and_scale_data(x, y, random_state=42):
    """
    Investigating
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=random_state, stratify=y, test_size=0.2
    )

    print("\nTraining set class distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTesting set class distribution:")
    print(y_test.value_counts(normalize=True))

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return (
        x_train_scaled,
        x_test_scaled,
        y_train_encoded,
        y_test_encoded,
        x_test,
        y_test,
    )


def train_and_evaluate_models(
    x_train, x_test, y_train, y_test, x_orig_test, y_orig_test
):
    """
    Investigating
    """

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # Helps with imbalanced classes
                C=0.1,  # Regularization to prevent overfitting
                solver="liblinear",  # Better for smaller datasets
            ),
            {
                "C": [0.001, 0.01, 0.1, 1, 10],  # Different regularization strengths
                "penalty": ["l1", "l2"],  # Different regularization types
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,  # More trees
                class_weight="balanced",
                max_depth=None,  # Allow deeper trees
                min_samples_split=2,
                min_samples_leaf=1,
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        ),
    }
    trained_models = {}
    for name, (model, param_grid) in models.items():

        print(f"\n--- {name} ---")

        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring="recall"  # Optimize for recall
        )
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)

        best_model.fit(x_train, y_train)

        y_pred = best_model.predict(x_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        y_scores = best_model.predict_proba(x_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        avg_precision = average_precision_score(y_test, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"{name} (AP = {avg_precision:.2f})")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

        trained_models[name] = best_model

    return trained_models


def export_models_and_metadata(trained_models, scaler, feature_columns):
    """
    Investigating
    """

    for name, model in trained_models.items():

        # Create a safe filename by replacing spaces with underscores
        model_filename = f'{name.replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")

    # Save the scaler
    joblib.dump(scaler, "feature_scaler.pkl")
    print("Feature scaler saved as feature_scaler.pkl")

    # Save feature column names
    feature_names_filename = "feature_columns.json"
    with open(feature_names_filename, "w") as f:
        json.dump(list(feature_columns), f)
    print(f"Feature column names saved as {feature_names_filename}")


def main():
    """
    Investigating
    """

    filepath = "data/csv/clean_data.csv"

    (
        x,
        y,
    ) = load_and_preprocess_data(filepath)

    (
        x_train_scaled,
        x_test_scaled,
        y_train_encoded,
        y_test_encoded,
        x_orig_test,
        y_orig_test,
    ) = split_and_scale_data(x, y)

    trained_models = train_and_evaluate_models(
        x_train_scaled,
        x_test_scaled,
        y_train_encoded,
        y_test_encoded,
        x_orig_test,
        y_orig_test,
    )


if __name__ == "__main__":
    main()
