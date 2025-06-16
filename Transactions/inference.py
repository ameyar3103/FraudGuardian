import pandas as pd
from joblib import load
from model import feature_engineering  # replace with your actual filename, e.g., fraud_train_script
import os

# 1. Load saved model
model_path = 'fraud_detection_pipeline.joblib'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
pipeline = load(model_path)
print("Loaded pipeline.")

# 2. Load test data
test_path = 'CreditCardDataset/fraudTest.csv'
df = pd.read_csv(test_path)

# 3. Feature engineering (reuse function from training)
df_fe, _ = feature_engineering(df)  # use new data without ref_freq_maps; or load/save maps if needed

# 4. Drop columns that were removed during training
drop_cols = [
    'cc_num', 'first', 'last', 'street', 'trans_date_trans_time', 'dob',
    'trans_dt', 'dob_dt', 'trans_num', 'zip'
]
df_fe.drop(columns=[col for col in drop_cols if col in df_fe.columns], inplace=True)

# 5. Select input features
numeric_features = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long',
                    'trans_hour', 'trans_dow', 'age', 'distance_km',
                    'merchant_freq', 'city_freq', 'job_freq']
categorical_features = ['category', 'gender', 'state']
X = df_fe[numeric_features + categorical_features]

# 6. Predict
print("Predicting...")
df['fraud_prediction'] = pipeline.predict(X)
df['fraud_probability'] = pipeline.predict_proba(X)[:, 1]

# 7. Show sample output
fraud_case = df[df['fraud_prediction'] == 1]
if not fraud_case.empty:
    print("\n=== Example Fraudulent Prediction ===")
    print(fraud_case.iloc[0][['amt', 'merchant', 'category','city_pop','lat','long','merch_lat','merch_long','gender','state', 'fraud_prediction', 'fraud_probability']])
else:
    print("No fraudulent predictions found in test set.")
