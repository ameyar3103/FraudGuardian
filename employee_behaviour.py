import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# === Step 1: Load New Input Data ===
input_file = "new_employees.csv"  # Replace with your input CSV
df_new = pd.read_csv(input_file)

# === Step 2: Preprocess Login/Logout time ===
def time_to_minutes(t):
    return int(datetime.strptime(t, "%H:%M").hour) * 60 + int(datetime.strptime(t, "%H:%M").minute)

df_new["LoginMinutes"] = df_new["LoginTime"].apply(time_to_minutes)
df_new["LogoutMinutes"] = df_new["LogoutTime"].apply(time_to_minutes)
df_new["WorkDuration"] = df_new["LogoutMinutes"] - df_new["LoginMinutes"]
df_new.drop(columns=["LoginTime", "LogoutTime"], inplace=True)

# === Step 3: Setup the same pipeline as training ===
categorical_features = ["EmployeeRole"]
numeric_features = df_new.drop(columns=categorical_features).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", SimpleImputer(strategy="mean"), numeric_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# === Step 4: Load Training Data (to fit pipeline) ===
df_train = pd.read_csv("employee_fraud_risk_dataset.csv")
df_train["LoginMinutes"] = df_train["LoginTime"].apply(time_to_minutes)
df_train["LogoutMinutes"] = df_train["LogoutTime"].apply(time_to_minutes)
df_train["WorkDuration"] = df_train["LogoutMinutes"] - df_train["LoginMinutes"]
df_train.drop(columns=["LoginTime", "LogoutTime"], inplace=True)

X_train = df_train.drop(columns=["FraudRiskScore"])
y_train = df_train["FraudRiskScore"]

# Fit the model
model.fit(X_train, y_train)

# === Step 5: Predict on new data ===
predictions = model.predict(df_new)
df_new["PredictedFraudRiskScore"] = predictions

# === Step 6: Save predictions ===
output_file = "predicted_fraud_risks.csv"
df_new.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
