import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Attempt to import SMOTE from imbalanced-learn; if unavailable, fall back to class_weight only
use_smote = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    use_smote = True
    print("imblearn available: will use SMOTE to oversample minority class.")
except ImportError:
    print("imbalanced-learn not installed; will proceed without SMOTE and rely on class_weight='balanced'.")


def load_data(train_path: str, test_path: str):
    """
    Load train and test CSVs into pandas DataFrames.
    """
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance (in kilometers) between two points arrays.
    Inputs can be Pandas Series or numpy arrays (in degrees).
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def feature_engineering(df: pd.DataFrame, ref_freq_maps=None):
    """
    Perform feature engineering on a DataFrame:
      - Parse transaction datetime, extract hour and day of week.
      - Parse date of birth, compute age at transaction time (with corrected timedelta logic).
      - Compute distance between transaction location and merchant location.
      - Frequency-encode high-cardinality columns: merchant, city, job.
        If ref_freq_maps is provided (dict of maps from train), use them; otherwise compute from df.
    Returns:
      - df_fe: DataFrame with new features added.
      - freq_maps: dict of frequency maps computed (only if ref_freq_maps was None).
    """
    df = df.copy()
    # 1. Parse transaction datetime
    if 'trans_date_trans_time' not in df.columns:
        raise KeyError("Column 'trans_date_trans_time' not found in DataFrame for feature_engineering.")
    df['trans_dt'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    # Drop rows where trans_dt is NaT
    n_before = len(df)
    df = df[~df['trans_dt'].isna()].copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows due to invalid trans_date_trans_time parsing.")

    # Extract hour and day of week
    df['trans_hour'] = df['trans_dt'].dt.hour
    df['trans_dow'] = df['trans_dt'].dt.dayofweek  # Monday=0, Sunday=6

    # 2. Parse date of birth and compute age at transaction
    # If 'dob' not in columns, age will be all NaN
    if 'dob' in df.columns:
        df['dob_dt'] = pd.to_datetime(df['dob'], errors='coerce')
        # Debug check (can uncomment if needed):
        # print("trans_dt dtype:", df['trans_dt'].dtype, "dob_dt dtype:", df['dob_dt'].dtype)
        # Compute age in years via subtraction of datetime64 Series
        df['age'] = (df['trans_dt'] - df['dob_dt']).dt.days / 365.25
        # Implausible ages → NaN
        df.loc[df['age'] < 0, 'age'] = np.nan
        df.loc[df['age'] > 120, 'age'] = np.nan
    else:
        df['age'] = np.nan
        print("Warning: 'dob' column not found; setting 'age' to NaN for all rows.")

    # 3. Distance between transaction location and merchant location
    if {'lat', 'long', 'merch_lat', 'merch_long'}.issubset(df.columns):
        df['distance_km'] = haversine_vectorized(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    else:
        df['distance_km'] = np.nan
        print("Warning: One of 'lat','long','merch_lat','merch_long' missing; setting 'distance_km' to NaN.")

    # 4. Frequency encoding for high-cardinality columns
    freq_maps = {}
    cols_to_encode = ['merchant', 'city', 'job']
    if ref_freq_maps is None:
        for col in cols_to_encode:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                df[f'{col}_freq'] = df[col].map(freq).fillna(0.0)
                freq_maps[col] = freq.to_dict()
            else:
                print(f"Warning: column '{col}' not in DataFrame; skipping frequency encoding.")
    else:
        for col in cols_to_encode:
            if col in df.columns:
                fmap = ref_freq_maps.get(col, {})
                df[f'{col}_freq'] = df[col].map(fmap).fillna(0.0)
                freq_maps[col] = fmap
            else:
                print(f"Warning: column '{col}' not in DataFrame; skipping frequency encoding (ref map provided).")

    # Return engineered DataFrame and freq_maps
    return df, freq_maps


def build_preprocessor(numeric_features, categorical_features):
    """
    Build a ColumnTransformer for preprocessing:
      - Numeric features: impute missing with median, then StandardScaler.
      - Categorical features: impute missing with 'missing', then OneHotEncoder(handle_unknown='ignore').
    Returns the ColumnTransformer.
    """
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])
    return preprocessor


def main():
    # Paths to your CSV files; adjust as needed
    train_path = 'CreditCardDataset/fraudTrain.csv'
    test_path  = 'CreditCardDataset/fraudTest.csv'

    # 1. Load data
    train_df, test_df = load_data(train_path, test_path)
    print((train_df["is_fraud"]==1).sum())
    print((test_df["is_fraud"]==1).sum())
    # 2. Feature engineering on train
    train_fe, freq_maps = feature_engineering(train_df, ref_freq_maps=None)

    # 3. Feature engineering on test, using freq maps from train
    test_fe, _ = feature_engineering(test_df, ref_freq_maps=freq_maps)

    # 4. Drop identifier/sensitive columns
    drop_cols = [
        'cc_num', 'first', 'last', 'street', 'trans_date_trans_time', 'dob',
        'trans_dt', 'dob_dt', 'trans_num', 'zip'
    ]
    for col in drop_cols:
        if col in train_fe.columns:
            train_fe = train_fe.drop(columns=[col])
        if col in test_fe.columns:
            test_fe = test_fe.drop(columns=[col])

    # 5. Select features for modeling
    # Numeric features: 
    numeric_features = []
    for col in ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long',
                'trans_hour', 'trans_dow', 'age', 'distance_km',
                'merchant_freq', 'city_freq', 'job_freq']:
        if col in train_fe.columns:
            numeric_features.append(col)

    # Categorical features:
    categorical_features = []
    for col in ['category', 'gender', 'state']:
        if col in train_fe.columns:
            categorical_features.append(col)

    print("Numeric features used:", numeric_features)
    print("Categorical features used:", categorical_features)

    # 6. Prepare X and y
    if 'is_fraud' not in train_fe.columns:
        raise KeyError("Training data must contain 'is_fraud' column.")
    X_train = train_fe[numeric_features + categorical_features]
    y_train = train_fe['is_fraud'].astype(int)

    X_test = test_fe[numeric_features + categorical_features]
    if 'is_fraud' in test_fe.columns:
        y_test = test_fe['is_fraud'].astype(int)
    else:
        y_test = None
        print("Warning: 'is_fraud' not in test data; skipping evaluation at the end.")

    # 7. Build preprocessing pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 8. Build full pipeline with classifier, handling imbalance
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

    # 9. Train
    print("Starting training...")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    # 10. Evaluate on test set (if labels available)
    if y_test is not None:
        print("Predicting on test set...")
        y_pred = pipeline.predict(X_test)
        # Try to get probabilities for ROC AUC
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
            print("Warning: classifier does not support predict_proba; skipping ROC AUC.")
        # Classification report
        print("\n=== Classification Report on Test Set ===")
        print(classification_report(y_test, y_pred, digits=4))
        # Confusion matrix
        print("\n=== Confusion Matrix on Test Set ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        # ROC AUC
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                print(f"\nROC AUC on Test Set: {auc:.4f}")
            except ValueError:
                print("Warning: Unable to compute ROC AUC (perhaps only one class present in y_test).")
    else:
        print("Skipping evaluation since 'is_fraud' not in test data.")

    # 11. (Optional) Save the trained pipeline for later use
    from joblib import dump
    dump(pipeline, 'fraud_detection_pipeline.joblib')
    print("Saved trained pipeline to 'fraud_detection_pipeline.joblib'.")


if __name__ == "__main__":
    main()
