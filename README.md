# 🚀 Loan Approval Prediction

Predict whether a loan application should be approved using the classic Loan Prediction dataset. This repository provides modular notebooks, reusable Python modules, and reproducible artifacts for data exploration, model building, tuning, and deployment.

---

## 🏆 Key Features

- **Fast Data Ingestion & Validation:**  
  Utilities for loading and validating data (`src/data_collection.py`, `src/artifacts.py`).

- **Modular Preprocessing:**  
  Handles missing values, categorical encoding, and train/test splits (`src/preprocessing.py`).

- **Exploratory Data Analysis:**  
  Univariate and bivariate analysis notebooks (`notebooks/01–04_*`).

- **Baseline Model Comparison:**  
  Evaluate seven classifiers with metrics and confusion matrices (`src/model_baseline_and_evaluation.py`).

- **Automated Hyperparameter Tuning:**  
  For Logistic Regression, Linear SVM, and Naive Bayes, with persisted model artifacts (`src/tuner.py`).

- **Deployment Helper:**  
  Aligns incoming payloads with the training schema before scoring (`src/model_deployment.py`).

- **Batch Notebook Execution:**  
  Automate the pipeline with Papermill (`src/run_notebooks.py`).

---

## 📁 Repository Structure

```text
Loan-Approval-Prediction/
├─ data/
│  └─ raw/
│     └─ loan_data_set.csv         # Original Kaggle-style dataset
├─ artifacts/                      # Created after running notebooks/scripts
│  ├─ 01_raw.parquet               # Cached raw dataframe
│  ├─ 02_clean.parquet             # Cleaned + encoded dataframe
│  ├─ expected_cols.pkl            # Training feature schema
│  ├─ tuned_model_summary.csv      # Hyperparameter tuning report
│  └─ *_best.pkl                   # Persisted best-performing estimator
├─ notebooks/                      # Jupyter notebooks for each pipeline stage
├─ runs/                           # Papermill outputs (ignored until generated)
├─ src/
│  ├─ config.py                    # Centralized paths and constants
│  ├─ data_collection.py           # CSV loading + preview helpers
│  ├─ preprocessing.py             # Cleaning, encoding, split utilities
│  ├─ eda_univariate.py            # Target distribution & data quality checks
│  ├─ eda_bivariate.py             # Pairwise visualizations and insights
│  ├─ feature_selection.py         # SelectKBest, chi², ANOVA feature ranking
│  ├─ model_baseline_and_evaluation.py
│  ├─ tuner.py                     # GridSearchCV with persistence
│  ├─ model_deployment.py          # Inference wrapper
│  └─ run_notebooks.py             # Papermill automation
├─ requirements.txt                # Dependency list
└─ README.md                       # Project documentation
```

---

## 📊 Dataset

- **File:** `data/raw/loan_data_set.csv`
- **Rows/Columns:** 614 rows, 13 columns (Loan_ID, demographics, credit history, loan details, target)
- **Target:** `Loan_Status` (converted to `Loan_Status_Y` after dummy encoding)
- **Source:** Kaggle “Loan Prediction Problem” dataset  
  *(Update `config.CSV_PATH` if you use your own data)*

---

## ⚙️ Environment Setup

> **Note:** The current `requirements.txt` is a placeholder. Install dependencies manually or update the file before sharing.

1. **Install Python 3.10+** (Anaconda recommended)
2. **Create and activate a virtual environment:**
    ```sh
    conda create -n loan-approval python=3.10
    conda activate loan-approval
    ```
3. **Install core packages:**
    ```sh
    pip install -U pandas numpy scikit-learn matplotlib seaborn joblib papermill jupyter ipykernel
    ```
4. **(Optional) For Parquet support:**
    ```sh
    pip install pyarrow
    ```
5. **(Optional) Save dependencies:**
    ```sh
    pip freeze > requirements.txt
    ```

---

## 🚦 Running the Workflow

### 1. Interactive Notebooks

Launch Jupyter Lab/Notebook and open the files under `notebooks/`:

- `01_data_collection.ipynb` – Inspect raw data and schema
- `02_preprocessing.ipynb` – Clean, encode, and split
- `03_eda_univariate_bivariate.ipynb` – Exploratory analysis
- `04_feature_selection.ipynb` – Feature scoring
- `05_model_baseline_and_evaluation.ipynb` – Compare baseline models
- `06_final_model.ipynb` – Hyperparameter tuning and artifact export
- `07_model_deployment.ipynb` – Inference walkthrough

Each notebook saves intermediates in `artifacts/` for reproducibility.

---

### 2. Command-Line Scripts

All reusable functions live under `src/`. Example usage:

```python
import pandas as pd
from src.config import CSV_PATH
from src.preprocessing import clean_data
from src.model_baseline_and_evaluation import run_baseline_and_evaluation
from src.tuner import run_tuning_and_evaluation

# Load + clean
df_raw = pd.read_csv(CSV_PATH)
df_clean = clean_data(df_raw)

# Baseline comparison
baseline_results = run_baseline_and_evaluation(df_clean, target="Loan_Status_Y")

# Hyperparameter tuning + artifact persistence
tuned_results, best_model, model_path = run_tuning_and_evaluation(df_clean, target="Loan_Status_Y")
print(tuned_results)
print(f"Best model saved to: {model_path}")
```

---

### 3. Batch Notebook Execution

For automated runs (e.g., scheduled retraining), use Papermill:

```sh
python -m src.run_notebooks
```
> Ensure the `aiml` IPython kernel referenced in `run_notebooks.py` exists, or adjust `kernel_name`.

---

## 🚀 Model Deployment

After tuning, use the generated `*_best.pkl` and `expected_cols.pkl` in `artifacts/` to score new applications:

```python
from src.model_deployment import predict

sample = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
}
preds, probs = predict([sample])
print(preds, probs)
```
> `predict` returns the hard class (0/1) and, when available, the approval probability.

---

## 📦 Artifacts and Outputs

- `artifacts/01_raw.parquet`, `artifacts/02_*` – Cached datasets
- `artifacts/*_best.pkl` – Tuned estimator
- `artifacts/expected_cols.pkl` – Training columns for inference
- `artifacts/tuned_model_summary.csv` – Tuning results

Delete the `artifacts/` folder to start fresh.

---

## 🛠️ Troubleshooting

- **Non-ASCII characters:**  
  If you see `�` in output, switch to UTF-8 (`chcp 65001`) or use WSL/Git Bash.
- **Papermill kernel issues:**  
  Install with `python -m ipykernel install --user --name aiml`.
- **Unexpected columns:**  
  `clean_data` re-runs dummy encoding; review if columns look wrong.

---

## 🚧 Next Steps

- Replace the placeholder `requirements.txt` with actual dependencies.
- Add automated tests for preprocessing and inference.
- Track experiment metrics with MLflow or Weights & Biases.
- Containerize the deployment helper (FastAPI/Flask + Docker).

---

Happy modeling!  
Open an issue or reach out if you need help extending the workflow.