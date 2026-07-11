import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and column structure
model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Data Science Salary Predictor", page_icon="💰")
st.title("💰 Data Science Salary Predictor")
st.write("Fill in your details below to get an estimated salary (in USD).")

# --- User inputs ---
experience_level = st.selectbox(
    "Experience Level",
    options=["Entry-level", "Mid-level", "Senior-level", "Executive-level"]
)

company_size = st.selectbox(
    "Company Size",
    options=["Small", "Medium", "Large"]
)

job_title = st.selectbox(
    "Job Title",
    options=['Analytics Engineer', 'Applied Scientist', 'Business Intelligence Analyst',
             'Business Intelligence Engineer', 'Data Analyst', 'Data Architect', 'Data Engineer',
             'Data Manager', 'Data Science', 'Data Science Manager', 'Data Scientist',
             'ML Engineer', 'Machine Learning Engineer', 'Machine Learning Scientist', 'Other',
             'Research Analyst', 'Research Engineer', 'Research Scientist']
)

employment_type = st.selectbox(
    "Employment Type",
    options=["Contract", "Freelance", "Full-time", "Part-time"]
)

work_model = st.selectbox(
    "Work Model",
    options=["Hybrid", "On-site", "Remote"]
)

employee_residence = st.selectbox(
    "Your Country of Residence",
    options=['Canada', 'France', 'Germany', 'India', 'Other', 'Spain', 'United Kingdom', 'United States']
)

company_location = st.selectbox(
    "Company Location",
    options=['Canada', 'Germany', 'India', 'Other', 'Spain', 'United Kingdom', 'United States']
)

work_year = st.slider("Work Year", min_value=2020, max_value=2025, value=2024)

# --- Build the input row ---
if st.button("Predict Salary"):
    # Start with all-zero row matching training columns
    input_dict = {col: 0 for col in model_columns}

    # Ordinal encodings
    experience_order = {'Entry-level': 0, 'Mid-level': 1, 'Senior-level': 2, 'Executive-level': 3}
    company_size_order = {'Small': 0, 'Medium': 1, 'Large': 2}

    input_dict['experience_level_encoded'] = experience_order[experience_level]
    input_dict['company_size_encoded'] = company_size_order[company_size]
    input_dict['work_year'] = work_year

    # One-hot fields — set the matching column to 1 if it exists.
    # If a selected category isn't in model_columns, it's the "dropped" reference
    # category (Analytics Engineer / Contract / Hybrid / Canada) — leaving all
    # related columns at 0 correctly represents that category.
    onehot_fields = {
        f'job_title_grouped_{job_title}': 1,
        f'employment_type_{employment_type}': 1,
        f'work_models_{work_model}': 1,
        f'employee_residence_grouped_{employee_residence}': 1,
        f'company_location_grouped_{company_location}': 1,
    }

    for col, val in onehot_fields.items():
        if col in input_dict:
            input_dict[col] = val

    # Build dataframe in the exact column order the model expects
    input_df = pd.DataFrame([input_dict])[model_columns]

    # Predict (remember: model was trained on log-transformed salary)
    pred_log = model.predict(input_df)[0]
    pred_salary = np.expm1(pred_log)

    st.success(f"### Estimated Salary: ${pred_salary:,.0f}")
    st.caption("This is an estimate based on historical data (2020–2024) and has inherent uncertainty, "
               "especially for high-earning roles and countries outside the US.")