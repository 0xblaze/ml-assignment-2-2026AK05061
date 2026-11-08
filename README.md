# Inbox Model Lab - Machine Learning Assignment 2

**BITS ID:** 2026AK05061

## a. Problem statement

Classify an email as spam or non-spam using numeric summaries of its word frequencies, character frequencies and capital-letter patterns. Implement and compare Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine on the same dataset and held-out test partition.

Evaluate each classifier using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). Provide an interactive Streamlit application that loads saved models and evaluates labelled test CSV files.

Both error types matter: false positives incorrectly flag legitimate email, while false negatives allow spam through. Spam-class F1 is used to rank the observed results, with MCC as the tie-breaker.

## b. Dataset description

**Dataset:** [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

Each row represents one historical email. The inputs are numeric email statistics. The original target column is named `is_spam` in this project: **0 = non-spam**, **1 = spam**.

| Property | Value |
| --- | --- |
| Original instances | 4,601 |
| Input features | 57 |
| Task | Binary classification |
| Missing input values | 0 |
| Missing target values | 0 |
| Exact duplicate complete rows removed | 391 |
| Records retained for the experiment | 4,210 |
| Training records | 3,367 |
| Held-out test records | 843 |

The dataset meets the minimum requirement of 12 input features and 500 instances.

| Feature group | Count | Examples |
| --- | --- | --- |
| Word-frequency features | 48 | Frequency of words such as free, money and business |
| Character-frequency features | 6 | Frequency of characters such as !, $ and # |
| Capital-run statistics | 3 | Average and longest capital-run length, plus total capital letters |

### Class distribution

| Partition | Non-spam (0) | Spam (1) | Total |
| --- | ---: | ---: | ---: |
| Original dataset | 2,788 | 1,813 | 4,601 |
| Training partition | 2,024 | 1,343 | 3,367 |
| Test partition | 507 | 336 | 843 |

### Data preparation and split

1. Preserve the original data snapshot in `data/spambase_raw.csv`.
2. Remove exact duplicate complete records, retaining 4,210 rows.
3. Group identical input-feature profiles together, including profiles with conflicting target labels.
4. Use the first split from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` as an approximately 80/20 training/test split.
5. Save the training partition, test partition, feature names, class counts and SHA-256 checksums.

There are **zero overlapping input profiles** between the training and test partitions. Only one held-out split is evaluated; the reported results are not five-fold cross-validation averages.

**Dataset attribution:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase*. UCI Machine Learning Repository. [DOI: 10.24432/C53G6X](https://doi.org/10.24432/C53G6X). Dataset licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## c. GitHub Repository Link

[GitHub repository](https://github.com/0xblaze/ml-assignment-2-2026AK05061)

[Live Streamlit application](https://ml-assignment-2-2026ak05061-txhki4h65gholiktgusa8c.streamlit.app/)

## d. Models used

All five classifiers use the same training and test partitions. Each saved artifact is a complete scikit-learn pipeline containing preprocessing and the classifier.

Median imputation is fitted on training data only. Logistic Regression, KNN, Gaussian Naive Bayes and SVM also use a training-fitted `StandardScaler`. Decision Tree uses imputation without scaling. Although this dataset has no missing values, the saved imputers also support missing feature cells during test-file evaluation.

| Model | Implementation and explicit settings |
| --- | --- |
| Logistic Regression | `LogisticRegression(max_iter=2000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42)` |
| KNN | `KNeighborsClassifier(n_neighbors=7, weights='distance')` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | `SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=128)` |

No hyperparameter search was performed. Test data is used for evaluation; preprocessing and classifiers are fitted using the training partition only.

### Evaluation metric comparison

The following results were obtained from the **843-row held-out test partition**, evaluating the saved model artifacts after reloading them.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9134 | 0.9608 | 0.9122 | 0.8661 | 0.8885 | 0.8186 |
| Decision Tree | 0.9134 | 0.9314 | 0.9228 | 0.8542 | 0.8872 | 0.8187 |
| KNN | 0.9063 | 0.9576 | 0.9003 | 0.8601 | 0.8798 | 0.8036 |
| Naive Bayes | 0.8114 | 0.9376 | 0.6945 | 0.9405 | 0.7990 | 0.6545 |
| Support Vector Machine | 0.9170 | 0.9677 | 0.9182 | 0.8690 | 0.8930 | 0.8261 |

Precision, Recall and F1 use **spam (1)** as the positive class. AUC uses continuous spam probabilities for Logistic Regression, Decision Tree, KNN and Naive Bayes, and the continuous decision function for SVM. Precision, Recall and F1 use `zero_division=0` when a denominator is undefined. For an uploaded test file containing only one class, the application displays AUC as N/A.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | A strong linear baseline, with 91.34% accuracy and F1 of 0.8885. It incorrectly flagged 28 legitimate emails and missed 45 spam emails, producing the second-highest F1 on this split. |
| Decision Tree | Achieved the highest spam precision, 0.9228, and the fewest false positives, 24. Its recall of 0.8542 was the lowest among the five models, with 49 missed spam emails. It therefore favoured fewer incorrect spam flags at the cost of catching less spam. |
| KNN | Achieved F1 of 0.8798 and AUC of 0.9576. It produced 32 false positives and 47 false negatives. With the selected seven distance-weighted neighbours, its F1 was below Logistic Regression, Decision Tree and SVM. |
| Naive Bayes | Achieved the highest recall, 0.9405, detecting 316 of 336 spam emails and missing only 20. However, it incorrectly flagged 139 legitimate emails, reducing precision to 0.6945 and F1 to 0.7990. Its observed trade-off favours catching spam over avoiding false alarms. |
| Support Vector Machine | Achieved the highest accuracy (0.9170), AUC (0.9677), F1 (0.8930) and MCC (0.8261). It produced 26 false positives and 44 false negatives, giving the strongest F1-based balance on this test partition. |
| Overall winner for this dataset experiment | **Support Vector Machine**, using the stated F1 ranking criterion. Decision Tree offers the highest precision and Naive Bayes offers the highest recall, so a deployment decision would also depend on the relative cost of the two error types. |

These observations describe this held-out experiment. A single split does not establish statistically significant differences or universal superiority. Spambase is historical and contains personalised non-spam indicators; these results do not establish performance on contemporary email traffic.

## Streamlit application

The application provides:

- Bundled held-out data for immediate evaluation.
- A download button for the labelled test CSV.
- CSV upload with all 57 input features and the `is_spam` target.
- A dropdown for selecting any of the five saved classifiers.
- All six evaluation metrics for the selected model.
- A confusion matrix and classification report.
- A comparison of all five classifiers on the active test data.
- Downloadable comparison results and per-row predictions.
- An ivory, forest-green and terracotta theme.

Uploaded data is evaluated using saved pipelines. **The app does not retrain models on uploads.** It validates column names, numeric values, binary target labels and infinite input values. Application-level limits are 20 MB and 50,000 rows per test CSV. Missing feature cells are handled by the saved imputer; missing target labels are rejected.

SVM prediction downloads contain decision scores, which are not probabilities. The other four models provide estimated spam probabilities.

## Repository files

| File or directory | Purpose |
| --- | --- |
| `app.py` | Streamlit evaluation frontend |
| `.streamlit/config.toml` | Custom application theme |
| `requirements.txt` | Pinned Python package versions |
| `inspect_spambase.py` | Download/cache and inspect the original dataset |
| `prepare_data.py` | Deduplication, grouped partitioning and metadata |
| `ml_utils.py` | Shared data validation and metric calculations |
| `train.py` | Train, save, reload and evaluate all five pipelines |
| `data/spambase_raw.csv` | Original dataset snapshot with the renamed target |
| `data/train_data.csv` | Training partition |
| `data/split_metadata.json` | Feature schema, split details and checksums |
| `test_data.csv` | Labelled held-out data used in the experiment |
| `model/*.joblib` | Five saved preprocessing/classifier pipelines |
| `model/manifest.json` | Model filenames, checksums, settings and environment versions |
| `reports/model_comparison.csv` | Six-metric comparison for the five classifiers |
| `reports/model_details.json` | Detailed metrics, confusion matrices, classification reports and fit durations |

## Run locally

Use Python **3.14** or **3.12**, matching the training environment.

```bash
git clone https://github.com/0xblaze/ml-assignment-2-2026AK05061.git
cd ml-assignment-2-2026AK05061
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The repository includes the saved models and test data needed to run the app.

### Reproduce training results

From the repository root, with the environment activated:

```bash
python inspect_spambase.py
python prepare_data.py
python train.py
```

The inspection script reuses the cached raw CSV when present. The preparation script records the shared split and checksums. Training checks those checksums before fitting and calculates the reported metrics from the reloaded saved models.

### Training environment

Development and training were performed in WSL using VS Code.

| Component | Version |
| --- | --- |
| Python | 3.14.4 |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| Matplotlib | 3.11.2 |
| Streamlit | 1.67.0 |
| ucimlrepo | 0.0.7 |

The complete dependency list is pinned in `requirements.txt`. Keep deployment dependencies consistent with those used to create the saved model files.

## Streamlit Community Cloud deployment

The live application is deployed from this GitHub repository, using branch `main` and entrypoint `app.py`. For a fresh deployment, select Python 3.14 in Advanced settings to match the training environment, and install dependencies from the committed `requirements.txt`.

Public access, all five model selections, the test CSV download, and re-upload of that CSV were checked on 8 November 2026. The live comparison matched the training results. For SVM, the confusion matrix contained **481 true negatives, 26 false positives, 44 false negatives and 292 true positives**.

