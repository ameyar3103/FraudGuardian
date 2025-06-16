import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# === Convert HH:MM to Minutes ===
def time_to_minutes(t):
    return int(datetime.strptime(t, "%H:%M").hour) * 60 + int(datetime.strptime(t, "%H:%M").minute)
# === Load + Train Model ===
@st.cache_data
def train_model():
    df_train = pd.read_csv("employee_fraud_risk_dataset.csv")
    df_train["LoginMinutes"] = df_train["LoginTime"].apply(time_to_minutes)
    df_train["LogoutMinutes"] = df_train["LogoutTime"].apply(time_to_minutes)
    df_train["WorkDuration"] = df_train["LogoutMinutes"] - df_train["LoginMinutes"]
    df_train.drop(columns=["LoginTime", "LogoutTime"], inplace=True)
    categorical_features = ["EmployeeRole"]
    numeric_features = df_train.drop(columns=["FraudRiskScore"] + categorical_features).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", SimpleImputer(strategy="mean"), numeric_features)
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    X_train = df_train.drop(columns=["FraudRiskScore"])
    y_train = df_train["FraudRiskScore"]
    model.fit(X_train, y_train)
    return model
# === Streamlit App ===
st.set_page_config(layout="wide")
st.title("Employee Fraud Risk Predictor & Dashboard")
uploaded_file = st.file_uploader("Upload the new employee data file (CSV)", type="csv")
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    try:
        # Preprocess
        df_new["LoginMinutes"] = df_new["LoginTime"].apply(time_to_minutes)
        df_new["LogoutMinutes"] = df_new["LogoutTime"].apply(time_to_minutes)
        df_new["WorkDuration"] = df_new["LogoutMinutes"] - df_new["LoginMinutes"]
        df_new.drop(columns=["LoginTime", "LogoutTime"], inplace=True)
        # Predict
        model = train_model()
        predictions = model.predict(df_new)
        df_new["PredictedFraudRiskScore"] = predictions
        df_new["Risk Score"] = (df_new["PredictedFraudRiskScore"] * 100).round(2)
        # Simulate alerts
        alerts = []
        for idx, row in df_new.iterrows():
            if row['FailedLogins_Daily'] > 2:
                alerts.append({'Date': 'June 12', 'Employee': f"{row['EmployeeRole']} {idx}", 'Alert': 'Multiple failed logins'})
            elif row['ManualOverrides_Daily'] > 3:
                alerts.append({'Date': 'June 11', 'Employee': f"{row['EmployeeRole']} {idx}", 'Alert': 'Frequent manual overrides'})
            elif row['WorkDuration'] > 700:
                alerts.append({'Date': 'June 10', 'Employee': f"{row['EmployeeRole']} {idx}", 'Alert': 'Unusual work duration'})
        alerts_df = pd.DataFrame(alerts)
        # Top employees
        top_employees = df_new.sort_values("Risk Score", ascending=False).head(5).copy()
        top_employees.reset_index(inplace=True)
        top_employees["Employee"] = top_employees["EmployeeRole"] + " " + top_employees["index"].astype(str)
        # UI Sections
        st.success("Fraud Risk Prediction Completed!")
        st.subheader(":page_facing_up: Predicted Data")
        st.dataframe(df_new)
        # Download
        csv = df_new.to_csv(index=False).encode('utf-8')
        st.download_button(":inbox_tray: Download Predicted Results", csv, "predicted_fraud_risks.csv", "text/csv")
        # Layout for dashboard
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(":bar_chart: Risk Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_new['Risk Score'], bins=10, ax=ax)
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Number of Employees")
            st.pyplot(fig)
        with col2:
            st.subheader(":rotating_light: Recent Alerts")
            if not alerts_df.empty:
                st.dataframe(alerts_df)
            else:
                st.write("No alerts triggered based on current thresholds.")
        col3, col4 = st.columns([1, 1])
        with col3:
            st.subheader(":rotating_light: Top 5 Risky Employees")
            st.dataframe(top_employees[["Employee", "EmployeeRole", "Risk Score"]])
        with col4:
            st.subheader(":pushpin: Feature Correlation Heatmap")
            num_cols = df_new.select_dtypes(include='number')
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
    except Exception as e:
        st.error(f"Error while processing file: {e}")






