import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
TARGET = "is_spam"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root=ROOT, seed=42):
    data_folder = root / "data"
    raw_path = data_folder / "spambase_raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError("Run inspect_spambase.py first.")

    raw = pd.read_csv(raw_path)

    if TARGET not in raw.columns:
        raise ValueError("The dataset must contain the is_spam target.")

    feature_names = [
        column for column in raw.columns if column != TARGET
    ]

    if len(feature_names) < 12 or len(raw) < 500:
        raise ValueError("Dataset does not meet assignment size requirements.")

    if raw[TARGET].isna().any():
        raise ValueError("Missing target labels were found.")

    if not raw[TARGET].isin([0, 1]).all():
        raise ValueError("Target labels must be 0 or 1.")

    # Remove repeated complete records while preserving the raw CSV.
    clean = raw.drop_duplicates().reset_index(drop=True)

    # Group identical inputs, including inputs with conflicting labels.
    groups = pd.util.hash_pandas_object(
        clean[feature_names],
        index=False,
    )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    # Use one held-out fold: approximately 80% training and 20% testing.
    train_index, test_index = next(
        splitter.split(
            clean[feature_names],
            clean[TARGET],
            groups,
        )
    )

    train = clean.iloc[train_index].reset_index(drop=True)
    test = clean.iloc[test_index].reset_index(drop=True)

    overlap = set(groups.iloc[train_index]) & set(groups.iloc[test_index])

    if overlap:
        raise ValueError("Identical input profiles overlap the partitions.")

    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("Both partitions must contain both classes.")

    train_path = data_folder / "train_data.csv"
    test_path = root / "test_data.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "source": "https://archive.ics.uci.edu/dataset/94/spambase",
        "dataset_citation": "Hopkins et al. (1999), Spambase, DOI: 10.24432/C53G6X",
        "dataset_license": "CC BY 4.0",
        "target": TARGET,
        "feature_names": feature_names,
        "seed": seed,
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "duplicates_removed": len(raw) - len(clean),
        "missing_input_cells": int(raw[feature_names].isna().sum().sum()),
        "train_rows": len(train),
        "test_rows": len(test),
        "split": "First fold of shuffled five-fold StratifiedGroupKFold",
        "train_class_counts": {
            str(label): int(count)
            for label, count in train[TARGET].value_counts().items()
        },
        "test_class_counts": {
            str(label): int(count)
            for label, count in test[TARGET].value_counts().items()
        },
        "raw_sha256": sha256(raw_path),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
    }

    metadata_path = data_folder / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Original rows:", len(raw))
    print("Duplicates removed:", metadata["duplicates_removed"])
    print("Remaining rows:", len(clean))
    print("Input features:", len(feature_names))
    print("Training rows:", len(train))
    print("Test rows:", len(test))
    print("Training class counts:", metadata["train_class_counts"])
    print("Test class counts:", metadata["test_class_counts"])
    print("Overlapping input profiles:", len(overlap))

    print("\nSaved:")
    print(train_path)
    print(test_path)
    print(metadata_path)


if __name__ == "__main__":
    prepare()