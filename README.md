#  Data Science Salary Predictor

A machine learning web app that predicts data science salaries (in USD) based on experience level, job title, location, company size, and other factors — trained on real-world salary data from 2020–2024.

**🔗 Live App:** [salary-predictor-project.streamlit.app](https://salary-predictor-project-jrhve7qsjz2lfxe.streamlit.app/)

---

## Overview

This project walks through a complete, end-to-end machine learning workflow: from raw data to a deployed, interactive web application. It predicts a data science professional's salary based on features like experience level, job title, remote work model, company size, and location.

**Problem type:** Regression
**Target variable:** `salary_in_usd`

---

## Dataset

- **Source:** [Data Science Salaries 2024 (Kaggle)](https://www.kaggle.com/datasets/sazidthe1/data-science-salaries)
- **Size:** 6,599 rows × 11 columns
- **Time span:** 2020–2024
- **Key features:** `job_title`, `experience_level`, `employment_type`, `work_models` (remote/hybrid/on-site), `company_size`, `employee_residence`, `company_location`, `work_year`

---

## Project Workflow

### 1. Exploratory Data Analysis
- Found salary distribution is **right-skewed** (most salaries cluster $80k–$200k, with a long tail toward $750k)
- Confirmed a clear positive relationship between `experience_level` and salary
- Identified `job_title` as **high-cardinality** (132 unique values), requiring grouping

### 2. Data Cleaning & Feature Engineering
- Grouped rare job titles, employee residences, and company locations into an "Other" bucket (threshold-based, to reduce sparsity while preserving signal)
- Applied **ordinal encoding** to `experience_level` and `company_size` (since they have a natural order)
- Applied **one-hot encoding** to nominal categorical features (`job_title`, `employment_type`, `work_models`, location fields)
- **Log-transformed** the target variable (`salary_in_usd`) to reduce the impact of right-skew and outliers on model training

### 3. Model Training & Comparison

Three models were trained and evaluated on a held-out test set (80/20 split):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $42,738 | $64,093 | 0.301 |
| Random Forest (default) | $43,246 | $63,871 | 0.306 |
| Random Forest (tuned via RandomizedSearchCV) | $42,762 | $64,066 | 0.302 |

**Key finding:** All three models converge to nearly identical performance (R² ≈ 0.30), regardless of algorithm complexity or hyperparameter tuning. This indicates the performance ceiling is driven by **feature richness, not model choice** — the dataset lacks granular signal (e.g., specific skills, exact years of experience, industry) needed to explain more salary variance.

### 4. Feature Importance

The strongest predictor by far was **`employee_residence: United States`**, followed by `experience_level`. This reflects genuine global salary disparities (US data roles pay substantially more in USD terms) rather than a subtle learned pattern.

### 5. Error Analysis

The model performs well in the common salary range ($50k–$250k) but **systematically underpredicts extreme high earners** (>$400k). This is a known limitation of tree-based averaging methods (like Random Forest) when very high values are rare in the training data — the model tends to regress toward the mean of similar-looking rows.

### 6. Deployment

The final tuned Random Forest model was saved with `joblib` and served through a **Streamlit** web app, allowing users to input their own details and get a live salary estimate.

---

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Linear Regression, Random Forest, RandomizedSearchCV)
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Model persistence:** joblib

---

## Limitations & Future Work

- **Feature richness:** The dataset doesn't include skills, exact years of experience, or industry — adding these would likely improve predictive power substantially.
- **High earner underprediction:** Consider quantile regression or a specialized model for the high-salary segment.
- **Job title granularity:** Titles were grouped into 18 buckets to manage cardinality; a more granular or embedding-based representation could capture more nuance.
- **Geographic bias:** The model leans heavily on US-vs-non-US as a signal; a cost-of-living-adjusted target could produce more globally comparable predictions.

---

## Run Locally

```bash
git clone https://github.com/Kavya132003/salary-predictor-project.git
cd salary-predictor-project
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Kavya — Data Science Trainee
[GitHub](https://github.com/Kavya132003)
