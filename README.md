# FinTech User Enrollment Predictor

A machine learning project that predicts whether a user will enroll in a financial product based on their app behavior and engagement data.

---

## Overview

This project analyzes user activity data from a fintech mobile app to build a **Logistic Regression classifier** that predicts user enrollment. It covers the full data science pipeline — from raw data cleaning to model evaluation.

---

## Project Structure

```
FinTech/
├── main.py               # Main script — full pipeline
├── data/
│   └── dataset.csv       # Raw user dataset
├── requirements.txt      # Project dependencies
└── README.md
```

---

## Pipeline

### 1. Data Loading & Exploration
- Load dataset with pandas
- Inspect column types, shapes, and missing values

### 2. Data Cleaning
- Parse datetime columns (`first_open`, `enrolled_date`) using `pd.to_datetime()`
- Compute time difference between enrollment and first app open (in hours)
- Handle `NaN` values gracefully

### 3. Exploratory Data Analysis
- Plot histograms of all numerical columns in a 3×3 grid
- Plot enrollment time difference histogram
- Use `tight_layout` for clean subplot spacing

### 4. Feature Engineering
- Extract savings screen interaction counts (`SavingsCount`) by summing relevant binary columns
- Drop raw screen columns after aggregation
- Remove non-numeric columns (e.g. `screen_list`) before scaling
- Separate `user` identifier and `enrolled` response variable

### 5. Preprocessing
- Split data into train/test sets (80/20)
- Apply `StandardScaler` to normalize features

### 6. Model Building
- **Model:** Logistic Regression (`penalty='l1'`, `solver='liblinear'`)
- Fit on training data
- Predict on test set

### 7. Evaluation
- Confusion matrix heatmap (Seaborn)
- Accuracy, Precision, Recall, F1 Score
- 10-fold cross-validation

---

## Results

| Metric | Score |
|---|---|
| Accuracy | _run model to generate_ |
| Precision | _run model to generate_ |
| Recall | _run model to generate_ |
| F1 Score | _run model to generate_ |
| CV Accuracy (mean ± std) | _run model to generate_ |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
python-dateutil
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Clone the repo
git clone https://github.com/your-username/fintech-enrollment-predictor.git
cd fintech-enrollment-predictor

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

## Key Concepts Used

- **Feature Engineering** — aggregating screen interaction columns into a single count feature
- **Standard Scaling** — normalizing features to zero mean and unit variance before logistic regression
- **L1 Regularization** — encourages sparsity, useful for high-dimensional behavioral data
- **Cross-Validation** — 10-fold CV to estimate real-world model performance
- **Confusion Matrix** — visual breakdown of true/false positives and negatives

---

## Notes

- `enrolled_date` contains missing values — rows with `NaT` are excluded from time-difference analysis
- The `user` column is preserved as an identifier but excluded from model training
- `screen_list` (raw comma-separated string column) is dropped before scaling

---

## Author

Built as part of a FinTech machine learning project.
