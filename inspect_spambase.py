from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"
data_folder.mkdir(exist_ok=True)
raw_file = data_folder / "spambase_raw.csv"

if raw_file.exists():
    emails = pd.read_csv(raw_file)
    print("Loaded local dataset.")
else:
    dataset = fetch_ucirepo(id=94)
    emails = dataset.data.features.copy()
    emails["is_spam"] = dataset.data.targets.iloc[:, 0]
    emails.to_csv(raw_file, index=False)
    print("Downloaded and saved the UCI dataset.")

features = emails.drop(columns="is_spam")
target = emails["is_spam"]

print("\nDATASET SIZE")
print("Emails:", len(emails))
print("Input features:", features.shape[1])

print("\nMISSING VALUES")
print("Inputs:", features.isna().sum().sum())
print("Target:", target.isna().sum())

print("\nCLASS COUNTS: 0 = non-spam, 1 = spam")
print(target.value_counts().sort_index())

print("\nCLASS PERCENTAGES")
print(target.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nDUPLICATE COMPLETE ROWS")
print(emails.duplicated().sum())

print("\nFIRST FIVE ROWS")
print(emails.head())

print("\nRaw dataset:", raw_file)