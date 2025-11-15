import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- LOAD FILES ----------------
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # list of column names used during training

# Load numeric columns EXACTLY as used during training
numeric_cols = scaler.feature_names_in_     # 🔥 AUTO-LOAD CORRECT ORDER

st.title("Loan Approval Prediction App 🚀")

# ---------------- USER INPUT FORM ----------------
st.header("Enter Applicant Details")

person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_income = st.number_input("Annual Income", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0)

gender = st.selectbox("Gender", ["male", "female"])
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master"])
loan_intent = st.selectbox("Loan Intent",
                            ["PERSONAL", "EDUCATION", "MEDICAL", "HOMEIMPROVEMENT",
                             "BUSINESS", "DEBTCONSOLIDATION"])

previous_default = st.selectbox("Previous Loan Default?", ["Yes", "No"])

# ---------------- COMPUTE loan_percent_income ----------------
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0

# ---------------- BUILD INPUT FRAME ----------------
input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "credit_score": credit_score,
    "person_emp_exp": person_emp_exp,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "loan_percent_income": loan_percent_income,  # 🔥 REQUIRED
    "person_gender": gender,
    "person_home_ownership": home,
    "person_education": education,
    "loan_intent": loan_intent,
    "previous_loan_defaults_on_file": previous_default
}

df = pd.DataFrame([input_data])

# ---------------- ONE-HOT ENCODING ----------------
df = pd.get_dummies(df)

# ---------------- ALIGN COLUMNS ----------------
df = df.reindex(columns=columns, fill_value=0)

# ---------------- SCALE NUMERIC COLUMNS (Auto-loaded) ----------------
df[numeric_cols] = scaler.transform(df[numeric_cols])

# ---------------- PREDICT ----------------
if st.button("Predict Loan Status"):
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- DASHBOARD SECTION ----------------
st.header("📊 Loan Dataset Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df_data = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")

    # ---------- 1️⃣ APPROVAL RATE PIE CHART ----------
    if "loan_status" in df_data.columns:
        st.subheader("Loan Approval Rate")

        approval_counts = df_data["loan_status"].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.pie(approval_counts, labels=approval_counts.index, autopct="%1.1f%%")
        ax1.axis("equal")
        st.pyplot(fig1)
    else:
        st.warning("Column 'loan_status' not found in dataset.")

    # ---------- 2️⃣ INCOME DISTRIBUTION ----------
    if "person_income" in df_data.columns:
        st.subheader("Income Distribution")

        fig2, ax2 = plt.subplots()
        ax2.hist(df_data["person_income"].dropna(), bins=30)
        ax2.set_xlabel("Income")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    else:
        st.warning("Column 'person_income' not found.")

    # ---------- 3️⃣ CREDIT SCORE DISTRIBUTION ----------
    if "credit_score" in df_data.columns:
        st.subheader("Credit Score Distribution")

        fig3, ax3 = plt.subplots()
        ax3.hist(df_data["credit_score"].dropna(), bins=30)
        ax3.set_xlabel("Credit Score")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)
    else:
        st.warning("Column 'credit_score' not found.")

    # ---------- 4️⃣ LOAN AMOUNT VS APPROVAL ----------
    if "loan_amnt" in df_data.columns and "loan_status" in df_data.columns:
        st.subheader("Loan Amount vs Approval Status")

        grouped = df_data.groupby("loan_status")["loan_amnt"].mean()

        fig4, ax4 = plt.subplots()
        ax4.bar(grouped.index, grouped.values)
        ax4.set_xlabel("Loan Status")
        ax4.set_ylabel("Average Loan Amount")
        st.pyplot(fig4)
    else:
        st.warning("Required columns not found.")
