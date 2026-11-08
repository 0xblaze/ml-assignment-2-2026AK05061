import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import evaluate_model, validate_data
from prepare_data import ROOT, sha256


def train():
    metadata_path = ROOT / "data" / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    train_path = ROOT / "data" / "train_data.csv"
    test_path = ROOT / "test_data.csv"

    # Ensure we are using the recorded, unchanged data partitions.
    if sha256(train_path) != metadata["train_sha256"]:
        raise ValueError("Training data changed. Run prepare_data.py again.")

    if sha256(test_path) != metadata["test_sha256"]:
        raise ValueError("Test data changed. Run prepare_data.py again.")

    feature_names = metadata["feature_names"]

    X_train, y_train = validate_data(
        pd.read_csv(train_path), feature_names
    )
    X_test, y_test = validate_data(
        pd.read_csv(test_path), feature_names
    )

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Training and test data must contain both classes.")

    models = {
        "Logistic Regression": (
            "logistic_regression",
            LogisticRegression(max_iter=2000),
        ),
        "Decision Tree": (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
        "KNN": (
            "knn",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
            ),
        ),
        "Naive Bayes": (
            "naive_bayes",
            GaussianNB(),
        ),
        "Support Vector Machine": (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                cache_size=128,
            ),
        ),
    }

    model_folder = ROOT / "model"
    report_folder = ROOT / "reports"
    model_folder.mkdir(exist_ok=True)
    report_folder.mkdir(exist_ok=True)

    packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "joblib", "matplotlib", "streamlit", "ucimlrepo",
    ]

    manifest = {
        "feature_names": feature_names,
        "target": metadata["target"],
        "python": platform.python_version(),
        "versions": {package: version(package) for package in packages},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_sha256": metadata["train_sha256"],
        "test_sha256": metadata["test_sha256"],
        "models": {},
    }

    comparison_rows = []
    detailed_results = {}

    for name, (filename, classifier) in models.items():
        print(f"\nTraining {name}...", flush=True)

        steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            )
        ]

        # Distance- and margin-based models benefit from scaled inputs.
        # GaussianNB also uses the same standardised feature representation.
        if name != "Decision Tree":
            steps.append(("scaler", StandardScaler()))

        steps.append(("classifier", classifier))
        pipeline = Pipeline(steps)

        # Fit preprocessing and classifier using training data only.
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        artifact_path = model_folder / f"{filename}.joblib"
        joblib.dump(pipeline, artifact_path, compress=3)

        # Evaluate the saved artifact, as the Streamlit app will.
        saved_model = joblib.load(artifact_path)
        result = evaluate_model(saved_model, X_test, y_test)

        comparison_rows.append({
            "Model": name,
            **result["metrics"],
        })

        detailed_results[name] = {
            "metrics": result["metrics"],
            "fit_seconds": fit_seconds,
            "confusion_matrix": result["confusion_matrix"].tolist(),
            "classification_report": result["classification_report"],
            "score_kind": result["score_kind"],
        }

        manifest["models"][name] = {
            "file": artifact_path.name,
            "sha256": sha256(artifact_path),
            "parameters": classifier.get_params(),
        }

        print(
            f"Saved {artifact_path.name}; "
            f"F1={result['metrics']['F1']:.4f}; "
            f"AUC={result['metrics']['AUC']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        report_folder / "model_comparison.csv",
        index=False,
    )

    (report_folder / "model_details.json").write_text(
        json.dumps(detailed_results, indent=2)
    )

    (model_folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nHELD-OUT TEST RESULTS")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved five trained pipelines in:", model_folder)
    print("Saved comparison and detailed reports in:", report_folder)


if __name__ == "__main__":
    train()