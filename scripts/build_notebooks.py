"""Build and execute the four analysis notebooks from the production code.

Usage (from the project root)::

    python scripts/build_notebooks.py

The script regenerates ``notebooks/*.ipynb`` (markdown + code cells) and
executes every notebook with the current virtual environment, so the
notebooks always stay in sync with ``src/`` and contain real outputs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BOOTSTRAP = (
    f"import sys\n"
    f"sys.path.insert(0, {str(PROJECT_ROOT)!r})  # make 'src' importable\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "%matplotlib inline\n"
)

KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
}


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


# ---------------------------------------------------------------------------
# Notebook 01 — Data exploration
# ---------------------------------------------------------------------------
def notebook_01() -> nbformat.NotebookNode:
    cells = [
        md(
            "# 01 — Data Exploration\n\n"
            "First contact with the raw **IBM Telco Customer Churn** dataset:\n"
            "schema, data types, summary statistics, missing values and value "
            "ranges. All loading logic is reused from `src/data_preprocessing.py` "
            "so the notebook stays in sync with the production code."
        ),
        code(BOOTSTRAP + "\n"
             "import pandas as pd\n"
             "from src.data_preprocessing import load_raw_data\n\n"
             "df = load_raw_data()\n"
             "print(f'Shape: {df.shape}')"),
        md("## 1. First look at the data"),
        code("df.head(8)"),
        md("## 2. Column names and data types"),
        code("pd.DataFrame({'dtype': df.dtypes, 'non_null': df.notna().sum()})"),
        md("## 3. Summary statistics (numerical columns)"),
        code("df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]"),
        md(
            "## 4. Missing values\n\n"
            "`TotalCharges` is stored as text and contains 11 blank values — "
            "these correspond to brand-new customers (`tenure == 0`)."
        ),
        code(
            "missing = (df.isna().sum()\n"
            "           + df.astype(str).apply(lambda s: s.str.strip() == '').sum())\n"
            "print(missing[missing > 0])"
        ),
        md("## 5. Value cardinality per categorical column"),
        code(
            "for col in df.select_dtypes(include='object').columns:\n"
            "    values = df[col].dropna().unique()\n"
            "    if len(values) <= 6:\n"
            "        print(f'{col:<22} {len(values):>2} values: {list(values)}')\n"
            "    else:\n"
            "        print(f'{col:<22} {len(values):>2} values '\n"
            "              f'(high cardinality, likely an ID)')"
        ),
        md(
            "**Key observations**\n\n"
            "* 7,043 rows, 21 columns — small, clean and realistic.\n"
            "* `customerID` is an identifier, not a feature.\n"
            "* `TotalCharges` is text and needs coercion; 11 blanks for new customers.\n"
            "* `SeniorCitizen` is stored as 0/1 while all other flags use Yes/No.\n"
            "* No other missing values — the dataset is mostly clean."
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=KERNEL_META)


# ---------------------------------------------------------------------------
# Notebook 02 — Data cleaning
# ---------------------------------------------------------------------------
def notebook_02() -> nbformat.NotebookNode:
    cells = [
        md(
            "# 02 — Data Cleaning\n\n"
            "Every cleaning step is **deterministic and row-wise** (no statistics "
            "learned from the data), so it is safe to apply before any split. "
            "Anything that requires learned statistics (imputation, scaling, "
            "encoding) lives inside the scikit-learn pipeline fitted on the "
            "training set only — see `src/data_preprocessing.py`."
        ),
        code(BOOTSTRAP + "\n"
             "import pandas as pd\n"
             "from src.data_preprocessing import (\n"
             "    clean_data, encode_target, load_raw_data,\n"
             ")\n"
             "from src.feature_engineering import add_features\n\n"
             "df = load_raw_data()"),
        md("## 1. The `TotalCharges` problem"),
        code(
            "blank = df['TotalCharges'].astype(str).str.strip().isin(['', 'nan'])\n"
            "print(f'Rows with blank TotalCharges: {blank.sum()}')\n"
            "df.loc[blank, ['tenure', 'MonthlyCharges', 'TotalCharges']]"
        ),
        md(
            "## 2. Cleaning steps\n\n"
            "1. strip column names;\n"
            "2. empty strings -> NaN;\n"
            "3. `TotalCharges` -> numeric (coerce);\n"
            "4. `SeniorCitizen` 0/1 -> \"No\"/\"Yes\";\n"
            "5. drop duplicates;\n"
            "6. blank `TotalCharges` with `tenure == 0` -> `MonthlyCharges` "
            "(a brand-new customer has no billing history yet)."
        ),
        code(
            "cleaned = clean_data(df)\n"
            "print(f'Shape after cleaning : {cleaned.shape}')\n"
            "print(f'Duplicates            : {cleaned.duplicated().sum()}')\n"
            "print(f'Missing values        : {cleaned.isna().sum().sum()}')"
        ),
        md("## 3. Verification of the TotalCharges fix"),
        code(
            "new = cleaned[cleaned['tenure'] == 0]\n"
            "assert new['TotalCharges'].notna().all()\n"
            "assert (new['TotalCharges'] == new['MonthlyCharges']).all()\n"
            "new[['tenure', 'MonthlyCharges', 'TotalCharges']]"
        ),
        md("## 4. Data types after cleaning"),
        code("cleaned.dtypes.to_frame('dtype')"),
        md("## 5. Target encoding"),
        code(
            "y = encode_target(cleaned)\n"
            "print(f'Churn rate: {y.mean():.1%}  (1 = churner)')"
        ),
        md("## 6. Feature engineering preview (details in notebook 03)"),
        code(
            "engineered = add_features(\n"
            "    cleaned.drop(columns=['Churn', 'customerID'])\n"
            ")\n"
            "engineered[['tenure', 'tenure_group', 'num_services',\n"
            "           'avg_monthly_spend']].head()"
        ),
        md(
            "**Outcome** — the cleaned dataset has 7,043 rows, 0 duplicates, 0 "
            "missing values and a 26.5% churn rate, ready for EDA and modelling."
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=KERNEL_META)


# ---------------------------------------------------------------------------
# Notebook 03 — EDA
# ---------------------------------------------------------------------------
EDA_SETUP = (
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from src import config\n"
    "from src.data_preprocessing import (\n"
    "    clean_data, encode_target, load_raw_data,\n"
    ")\n"
    "from src.feature_engineering import add_features\n\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n\n"
    "df = clean_data(load_raw_data())\n"
    "df['Churn'] = encode_target(df)\n"
    "df['Churn_label'] = np.where(df['Churn'] == 1, 'Yes', 'No')\n"
    "engineered = add_features(df.drop(columns=['customerID']))\n"
    "engineered['Churn'] = df['Churn'].values\n"
    "print(f'rows={len(df)}, churn rate={df[\"Churn\"].mean():.1%}')"
)


def notebook_03() -> nbformat.NotebookNode:
    cells = [
        md(
            "# 03 — Exploratory Data Analysis\n\n"
            "Business questions this notebook answers:\n\n"
            "* How imbalanced is churn?\n"
            "* Which customer segments have the highest churn?\n"
            "* Does contract type influence churn?\n"
            "* Does customer tenure relate to churn?\n"
            "* Are higher charges associated with churn?\n"
            "* Which services are associated with higher churn?\n\n"
            "Every figure is saved to `reports/figures/` for the README."
        ),
        code(BOOTSTRAP + EDA_SETUP),
        md("## 1. Churn distribution"),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "sns.countplot(data=df, x='Churn_label', ax=ax,\n"
            "              order=['No', 'Yes'])\n"
            "for bar in ax.patches:\n"
            "    ax.annotate(f'{bar.get_height():,}'\n"
            "                f'  ({bar.get_height()/len(df):.1%})',\n"
            "                (bar.get_x() + bar.get_width()/2, bar.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "ax.set_title('Churn distribution')\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_churn_distribution.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## 2. Does tenure relate to churn?\n\n"
            "Churn risk is concentrated in the first months of the relationship."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "sns.kdeplot(data=df, x='tenure', hue='Churn_label',\n"
            "            fill=True, alpha=0.35, common_norm=False, ax=ax)\n"
            "ax.set_title('Tenure distribution by churn status')\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_tenure_churn.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## 3. Does contract type influence churn?"),
        code(
            "ct = pd.crosstab(df['Contract'], df['Churn_label'],\n"
            "                 normalize='index') * 100\n"
            "ax = ct[['No', 'Yes']].plot(kind='bar', stacked=True,\n"
            "                          figsize=(7, 4.5),\n"
            "                          color=['#2ca02c', '#d62728'])\n"
            "ax.set_ylabel('% of customers')\n"
            "ax.set_title('Churn by contract type')\n"
            "for c in ax.containers:\n"
            "    ax.bar_label(c, fmt='%.0f%%', label_type='center')\n"
            "fig = ax.get_figure()\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_churn_by_contract.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "print('Churn rate by contract:')\n"
            "print(df.groupby('Contract')['Churn'].mean().round(3).to_string())"
        ),
        md("## 4. Payment method vs churn"),
        code(
            "ct = pd.crosstab(df['PaymentMethod'], df['Churn_label'],\n"
            "                 normalize='index') * 100\n"
            "ax = ct[['No', 'Yes']].plot(kind='barh', stacked=True,\n"
            "                          figsize=(8, 4.5),\n"
            "                          color=['#2ca02c', '#d62728'])\n"
            "ax.set_xlabel('% of customers')\n"
            "ax.set_title('Churn by payment method')\n"
            "fig = ax.get_figure()\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_churn_by_payment.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "print('Churn rate by payment method:')\n"
            "print(df.groupby('PaymentMethod')['Churn'].mean().round(3).to_string())"
        ),
        md("## 5. Internet service & subscribed services"),
        code(
            "ct = pd.crosstab(df['InternetService'], df['Churn_label'],\n"
            "                 normalize='index') * 100\n"
            "ax = ct[['No', 'Yes']].plot(kind='bar', stacked=True,\n"
            "                          figsize=(6.5, 4.5),\n"
            "                          color=['#2ca02c', '#d62728'])\n"
            "ax.set_ylabel('% of customers')\n"
            "ax.set_title('Churn by internet service')\n"
            "fig = ax.get_figure()\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_churn_by_internet.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "print('Churn rate by internet service:')\n"
            "print(df.groupby('InternetService')['Churn'].mean().round(3).to_string())"
        ),
        md("## 6. Are higher charges associated with churn?"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn_label',\n"
            "            fill=True, alpha=0.35, common_norm=False, ax=axes[0])\n"
            "axes[0].set_title('Monthly charges by churn status')\n"
            "sns.kdeplot(data=df, x='TotalCharges', hue='Churn_label',\n"
            "            fill=True, alpha=0.35, common_norm=False, ax=axes[1])\n"
            "axes[1].set_title('Total charges by churn status')\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_charges_by_churn.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## 7. Correlation structure of the numerical features"),
        code(
            "numeric = engineered[['tenure', 'MonthlyCharges', 'TotalCharges',\n"
            "                     'num_services', 'avg_monthly_spend', 'Churn']]\n"
            "fig, ax = plt.subplots(figsize=(7, 6))\n"
            "sns.heatmap(numeric.corr(), annot=True, fmt='.2f',\n"
            "            cmap='coolwarm', vmin=-1, vmax=1, ax=ax)\n"
            "ax.set_title('Correlation heatmap (numerical features)')\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_correlation_heatmap.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## 8. Churn rate by tenure group (engineered feature)"),
        code(
            "rate = engineered.groupby('tenure_group', observed=True)['Churn'].mean()\n"
            "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
            "sns.barplot(x=rate.index, y=rate.values * 100, ax=ax)\n"
            "for i, v in enumerate(rate.values * 100):\n"
            "    ax.text(i, v, f'{v:.0f}%', ha='center', va='bottom')\n"
            "ax.set_ylabel('Churn rate (%)')\n"
            "ax.set_title('Churn rate by tenure group')\n"
            "fig.savefig(config.FIGURES_DIR / 'eda_churn_by_tenure_group.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## 9. Interactive view (Plotly)"),
        code(
            "import plotly.express as px\n\n"
            "pivot = engineered.pivot_table(\n"
            "    index='Contract', columns='PaymentMethod',\n"
            "    values='Churn', aggfunc='mean',\n"
            ") * 100\n"
            "px.imshow(\n"
            "    pivot.round(1), text_auto='.1f',\n"
            "    color_continuous_scale='Reds', aspect='auto',\n"
            "    title='Churn rate (%) by contract type and payment method',\n"
            "    labels=dict(x='Payment method', y='Contract type', color='Churn %'),\n"
            ")"
        ),
        md(
            "## Key insights\n\n"
            "* **Contract type is the strongest signal**: ~43% of month-to-month "
            "customers churn vs ~3% of two-year contracts.\n"
            "* **Tenure**: churn risk collapses after the first 12 months.\n"
            "* **Charges**: higher monthly charges (fiber-optic bundles) are "
            "associated with churn; total charges mainly mirror tenure.\n"
            "* **Payment**: electronic-check payers churn much more than "
            "automatic credit-card payers.\n"
            "* **Internet**: fiber-optic customers churn more than DSL or "
            "no-internet customers.\n\n"
            "These patterns are exactly what the engineered features "
            "(tenure groups, service count, average monthly spend) aim to "
            "capture for the models."
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=KERNEL_META)


# ---------------------------------------------------------------------------
# Notebook 04 — Modelling & evaluation
# ---------------------------------------------------------------------------
def notebook_04() -> nbformat.NotebookNode:
    cells = [
        md(
            "# 04 — Modelling & Evaluation\n\n"
            "This notebook drives the production training script "
            "(`src.train.main`) end-to-end and displays the resulting "
            "artefacts: cross-validation comparison, threshold tuning, "
            "held-out test metrics, ROC/PR curves and SHAP explanations.\n\n"
            "**Strategy**\n"
            "* stratified 60/20/20 train/validation/test split;\n"
            "* 5-fold stratified cross-validation of Logistic Regression, "
            "Random Forest and XGBoost;\n"
            "* class imbalance handled with class weights / "
            "`scale_pos_weight` (no SMOTE — see README);\n"
            "* decision threshold tuned on the validation set (F1);\n"
            "* final evaluation on the untouched test set."
        ),
        code(BOOTSTRAP + "\n"
             "import pandas as pd\n"
             "from src import config\n"
             "from src.train import main\n\n"
             "# cleaning -> feature engineering -> CV -> selection ->\n"
             "# threshold tuning -> final training -> evaluation -> SHAP\n"
             "metrics = main()"),
        md("## 1. Cross-validation comparison (5-fold, stratified)"),
        code("pd.DataFrame(metrics['cv_table']).round(4)"),
        md("## 2. Model selection & decision threshold"),
        code(
            "print('Selected model :', metrics['model_name'])\n"
            "print('Selection rule :', metrics['selection_rule'])\n"
            "print('Best threshold :', metrics['threshold'],\n"
            "      f\"(maximising {metrics['threshold_metric']})\")\n"
            "print('Validation metrics at tuned threshold:',\n"
            "      {k: round(v, 3) for k, v in\n"
            "       metrics['validation_metrics_at_tuned_threshold'].items()})"
        ),
        md("## 3. Test-set evaluation (held out, never used during training)"),
        code(
            "results = pd.DataFrame([\n"
            "    {'scenario': 'threshold = 0.50',\n"
            "     **metrics['test_metrics_threshold_050']},\n"
            "    {'scenario': f\"tuned threshold = {metrics['threshold']}\",\n"
            "     **metrics['test_metrics_tuned_threshold']},\n"
            "])\n"
            "cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n"
            "results.set_index('scenario')[cols].round(4)"
        ),
        md("## 4. Diagnostic figures"),
        code(
            "from IPython.display import Image, display\n\n"
            "for name in ['confusion_matrix_test.png', 'roc_curves.png',\n"
            "             'pr_curve.png', 'threshold_curve.png']:\n"
            "    print(name)\n"
            "    display(Image(str(config.FIGURES_DIR / name)))"
        ),
        md("## 5. SHAP — global explanations"),
        code(
            "for name in ['shap_importance.png', 'shap_summary.png']:\n"
            "    print(name)\n"
            "    display(Image(str(config.FIGURES_DIR / name)))"
        ),
        md("## 6. Individual prediction & explanation"),
        code(
            "from src.predict import explain_prediction, predict_customer\n\n"
            "customer = {\n"
            "    'gender': 'Female', 'SeniorCitizen': 'No', 'Partner': 'No',\n"
            "    'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes',\n"
            "    'MultipleLines': 'No', 'InternetService': 'Fiber optic',\n"
            "    'OnlineSecurity': 'No', 'OnlineBackup': 'No',\n"
            "    'DeviceProtection': 'No', 'TechSupport': 'No',\n"
            "    'StreamingTV': 'Yes', 'StreamingMovies': 'No',\n"
            "    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',\n"
            "    'PaymentMethod': 'Electronic check',\n"
            "    'MonthlyCharges': 95.2, 'TotalCharges': 190.4,\n"
            "}\n"
            "result = predict_customer(customer)\n"
            "print(f\"Prediction: {result['prediction']}\")\n"
            "print(f\"Churn probability: {result['churn_probability']:.1%}\")\n"
            "print(f\"Confidence: {result['confidence']:.1%}\")\n\n"
            "explanation = explain_prediction(customer)\n"
            "print('Top drivers:')\n"
            "for _, row in explanation['table'].head(8).iterrows():\n"
            "    print(f\"  {row['feature']:>18} = {str(row['value']):<24} \"\n"
            "          f\"-> {row['direction']} risk\")"
        ),
        md(
            "## Conclusion\n\n"
            "* All three models clearly beat a majority-class baseline; the "
            "comparison table above reports the actual cross-validation scores.\n"
            "* The selected model balances ROC-AUC, churn recall and "
            "interpretability (see the selection rule).\n"
            "* The tuned threshold improves the precision/recall trade-off for "
            "a retention campaign.\n"
            "* The saved pipeline in `models/` powers `src/predict.py` and the "
            "Streamlit application."
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata=KERNEL_META)


# ---------------------------------------------------------------------------
# Build & execute everything
# ---------------------------------------------------------------------------
def build_all() -> None:
    """Generate and execute every notebook in ``notebooks/``."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01_data_exploration.ipynb": notebook_01(),
        "02_data_cleaning.ipynb": notebook_02(),
        "03_eda.ipynb": notebook_03(),
        "04_modeling.ipynb": notebook_04(),
    }

    for filename, nb in notebooks.items():
        path = NOTEBOOKS_DIR / filename
        nbformat.write(nb, path)
        print(f"\nExecuting {filename} ...")
        started = time.time()
        client = NotebookClient(
            nb, timeout=2400, kernel_name="python3", cwd=PROJECT_ROOT
        )
        client.execute()
        nbformat.write(nb, path)
        print(f"  -> done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    build_all()

