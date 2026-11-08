import io
import json
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from ml_utils import METRICS, evaluate_model, validate_data
from prepare_data import ROOT, sha256


st.set_page_config(
    page_title="Inbox Model Lab",
    page_icon="📬",
    layout="wide",
)


@st.cache_resource
def load_saved_models(manifest_text):
    """Cache repository-owned models, refreshing when the manifest changes."""
    manifest = json.loads(manifest_text)
    models = {}

    for name, entry in manifest["models"].items():
        path = ROOT / "model" / entry["file"]

        if path.parent.resolve() != (ROOT / "model").resolve():
            raise ValueError("Invalid saved-model path.")

        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Model checksum mismatch: {entry['file']}")

        models[name] = joblib.load(path)

    return models


st.markdown(
    """
    <div style="background:#183D32;border-left:6px solid #A8442D;
    padding:26px 30px;border-radius:16px;color:white;margin-bottom:22px">
        <div style="font-size:13px;letter-spacing:2px">
            INBOX MODEL LAB
        </div>
        <h1 style="color:white;margin:8px 0">
            Five models. One spam-detection task.
        </h1>
        <div>
            Explore the trade-off between catching spam
            and protecting legitimate email.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

manifest_path = ROOT / "model" / "manifest.json"

if not manifest_path.exists():
    st.error("Saved models are missing. Run train.py first.")
    st.stop()

try:
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)

    if manifest["versions"]["scikit-learn"] != version("scikit-learn"):
        raise ValueError(
            "The scikit-learn version differs from training. "
            "Install the project's pinned requirements."
        )

    models = load_saved_models(manifest_text)

except Exception as error:
    st.error(f"Unable to load saved models: {error}")
    st.stop()


st.sidebar.header("Evaluate test data")

source = st.sidebar.radio(
    "Test data source",
    ["Bundled held-out data", "Upload CSV"],
)

selected_model = st.sidebar.selectbox(
    "Classifier",
    list(models),
)

test_path = ROOT / "test_data.csv"

if test_path.exists():
    st.sidebar.download_button(
        "Download test CSV",
        data=test_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.sidebar.caption(
    "CSV requires all 57 input features and is_spam: "
    "0 = non-spam, 1 = spam."
)
st.sidebar.caption(
    "Models are already trained. Uploaded data is used for evaluation."
)

if source == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload labelled test data",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a test CSV or select bundled held-out data.")
        st.stop()

    content = uploaded_file.getvalue()
    source_name = uploaded_file.name

else:
    if not test_path.exists():
        st.error("The bundled test_data.csv is missing.")
        st.stop()

    content = test_path.read_bytes()
    source_name = "Bundled held-out test_data.csv"

try:
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Use a test CSV smaller than 20 MB.")

    frame = pd.read_csv(io.BytesIO(content))

    X, y = validate_data(
        frame,
        manifest["feature_names"],
        max_rows=50000,
    )

except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not use this CSV: {error}")
    st.stop()


st.caption(
    f"Active data: {source_name} | "
    f"{len(frame):,} emails | "
    f"{len(manifest['feature_names'])} input features"
)

if y.nunique() < 2:
    st.warning(
        "This file contains one class. "
        "AUC is undefined and shown as N/A."
    )

results = {
    name: evaluate_model(model, X, y)
    for name, model in models.items()
}

result = results[selected_model]

st.subheader(selected_model)

metric_columns = st.columns(3)

for index, metric in enumerate(METRICS):
    value = result["metrics"][metric]
    displayed_value = "N/A" if pd.isna(value) else f"{value:.4f}"
    metric_columns[index % 3].metric(metric, displayed_value)


left, right = st.columns([1, 1.2])

with left:
    st.subheader("Confusion matrix")

    figure, axis = plt.subplots(figsize=(5, 4))
    figure.patch.set_facecolor("#F7F4ED")
    axis.set_facecolor("#F7F4ED")

    chart_palette = matplotlib.colors.LinearSegmentedColormap.from_list(
        "inbox_palette",
        ["#F7F4ED", "#183D32"],
    )

    ConfusionMatrixDisplay(
        result["confusion_matrix"],
        display_labels=["Non-spam", "Spam"],
    ).plot(
        ax=axis,
        cmap=chart_palette,
        colorbar=False,
        values_format="d",
    )

    axis.xaxis.label.set_color("#263B32")
    axis.yaxis.label.set_color("#263B32")
    axis.tick_params(colors="#263B32")

    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    matrix = result["confusion_matrix"]

    st.caption(
        f"Legitimate emails marked as spam: {matrix[0, 1]:,}. "
        f"Spam emails missed: {matrix[1, 0]:,}."
    )

with right:
    st.subheader("Classification report")

    report_rows = {
        label: values
        for label, values in result["classification_report"].items()
        if isinstance(values, dict)
    }

    report = pd.DataFrame.from_dict(report_rows, orient="index")
    report["support"] = report["support"].astype(int)

    st.dataframe(report.round(4), width="stretch")

    st.caption(
        "Precision: how often a spam prediction is correct. "
        "Recall: how much of the actual spam was caught."
    )


st.subheader("Compare all five models on the active test data")

comparison = pd.DataFrame(
    [
        {"Model": name, **evaluation["metrics"]}
        for name, evaluation in results.items()
    ]
)

comparison = comparison.sort_values(
    ["F1", "MCC"],
    ascending=False,
).reset_index(drop=True)

st.dataframe(
    comparison.set_index("Model").round(4),
    width="stretch",
)

st.caption("Sorted by spam F1, then MCC.")

st.download_button(
    "Download comparison",
    data=comparison.to_csv(index=False),
    file_name="active_model_comparison.csv",
    mime="text/csv",
)


with st.expander("Inspect predictions"):
    predictions = frame.copy()
    predictions["predicted_is_spam"] = result["predictions"]
    predictions["spam_score"] = result["scores"]

    st.caption(result["score_kind"])
    st.dataframe(predictions.head(100), width="stretch")

    st.download_button(
        "Download predictions",
        data=predictions.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )


with st.expander("Dataset and evaluation method"):
    st.markdown(
        """
        **Dataset:** UCI Spambase, containing 4,601 historical emails
        and 57 numeric input features.

        Exact duplicate records were removed. Identical input profiles
        were kept together when creating the shared train/test split.

        Imputation and applicable scaling were fitted on training data
        and saved inside each model pipeline.

        AUC uses continuous scores: spam probabilities for four models
        and decision scores for SVM.

        This historical dataset includes personalised email indicators;
        its results do not establish performance on contemporary email.

        [UCI dataset source](https://archive.ics.uci.edu/dataset/94/spambase)
        """
    )