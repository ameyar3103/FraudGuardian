import streamlit as st
from joblib import load
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from Transactions.model import feature_engineering
import time  # for simulation delay

st.set_page_config(page_title="SecureBank Dashboard", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; font-size: 48px;'>SecureBank</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# Use an expander or directly show the form
with st.expander("Initiate a Transaction", expanded=True):
    st.markdown("### Enter Transaction Details")
    with st.form("transaction_form"):
        to_account         = st.text_input("To:", value="customer0")
        from_account       = st.text_input("From:", value="Customer1")
        transaction_time   = st.text_input("Time (YYYY-MM-DD HH:MM:SS):", value="2020-06-21 18:08:47")
        city_population    = st.number_input("City Population:", value=128715, step=1)
        latitude           = st.number_input("Latitude:", value=43.2326, format="%.6f")
        longitude          = st.number_input("Longitude:", value=-86.2492, format="%.6f")
        amount             = st.number_input("Amount (₹):", value=981.92, format="%.2f")
        category           = st.selectbox("Category:", ["shopping_net", "utility", "transfer", "salary"], index=0)
        gender             = st.selectbox("Gender:", ["M", "F"], index=0)
        state              = st.selectbox("State:", ["MI", "NY", "CA", "TX", "FL", "Other"], index=0)
        merchant_latitude  = st.number_input("Merchant Latitude:", value=43.849101, format="%.6f")
        merchant_longitude = st.number_input("Merchant Longitude:", value=-85.560458, format="%.6f")
        behavioral_feature_vec = st.text_input("Behavioral Feature Vector (comma-separated):",value="200,2,50,5,5,5,5,220")
        submitted = st.form_submit_button("Submit Transaction")
    # End of form

    if submitted:
        # Display a “Processing…” message immediately if desired
        st.info("Processing transaction...")

        # Wrap prediction in try/except
        try:
            # Load pipeline (ensure path is correct)
            pipeline = load('./Transactions/fraud_detection_pipeline.joblib')

            test_path = 'Transactions/fraudTest.csv'
            df = pd.read_csv(test_path)
            # df = pd.DataFrame([data_point])

            # 3. Feature engineering
            df_fe, _ = feature_engineering(df)

            # 4. Drop columns removed during training
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
            # Ensure all required features exist in df_fe; else KeyError
            missing_feats = [feat for feat in numeric_features + categorical_features if feat not in df_fe.columns]
            if missing_feats:
                st.error(f"Missing features after feature engineering: {missing_feats}")
                st.stop()

            X = df_fe[numeric_features + categorical_features]

            # 6. Predict (extract scalar)
            df['fraud_prediction'] = pipeline.predict(X)
            df['fraud_probability'] = pipeline.predict_proba(X)[:, 1]

            # 7. Show sample output
            fraud_case = df[df['fraud_prediction'] == 1]
            # 7. Display results immediately
            st.success("✅ Transaction Evaluated")
            if df['fraud_prediction'].iloc[1044] == 1:
                prediction = 1
                probability = df['fraud_probability'].iloc[1044]
                label1 = "Fraud" if prediction == 1 else "Not Fraud"
                emoji = "🔴" if prediction == 1 else "🟢"
            st.markdown(f"**Prediction:** {emoji} {label1}")
            st.markdown(f"**Fraud Probability:** {probability:.2%}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        np.random.seed(42)
        n_history = 200
        cluster_A = np.column_stack([
            np.random.poisson(lam=20, size=n_history // 2),
            np.random.exponential(scale=40, size=n_history // 2),
            np.random.uniform(300, 2000, size=n_history // 2),
            np.random.poisson(lam=0.2, size=n_history // 2),
            np.random.poisson(lam=1, size=n_history // 2),
            np.random.poisson(lam=1, size=n_history // 2),
            np.random.poisson(lam=1, size=n_history // 2),
            np.random.poisson(lam=25, size=n_history // 2)
        ])
        cluster_B = np.column_stack([
            np.random.poisson(lam=50, size=n_history // 2),
            np.random.exponential(scale=10, size=n_history // 2),
            np.random.uniform(1000, 3000, size=n_history // 2),
            np.random.poisson(lam=1, size=n_history // 2),
            np.random.poisson(lam=2, size=n_history // 2),
            np.random.poisson(lam=2, size=n_history // 2),
            np.random.poisson(lam=2, size=n_history // 2),
            np.random.poisson(lam=60, size=n_history // 2)
        ])
        history = np.vstack([cluster_A, cluster_B])
        columns = [
            'clicks_last_hour', 'avg_time_between_clicks', 'session_length',
            'num_failed_logins', 'device_change_rate', 'location_variance',
            'browser_jump_freq', 'actions_per_session'
        ]
        df_history = pd.DataFrame(history, columns=columns)

        # (Optional) New point(s) to check; replace with your actual new data for this user
        df_new = pd.DataFrame([
            {
                'clicks_last_hour': 200,
                'avg_time_between_clicks': 2,
                'session_length': 50,
                'num_failed_logins': 5,
                'device_change_rate': 5,
                'location_variance': 5,
                'browser_jump_freq': 5,
                'actions_per_session': 220
            }
        ])

        # ---------------------------------------------------
        # 2. Standardize historical features
        # ---------------------------------------------------
        scaler = StandardScaler()
        X_hist_scaled = scaler.fit_transform(df_history)  # shape (n_history, n_features)

        # ---------------------------------------------------
        # 3. Fit Gaussian Mixture Model on historical data
        # ---------------------------------------------------
        n_components = 2  # or choose via BIC/AIC
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X_hist_scaled)

        # ---------------------------------------------------
        # 4. Compute log-likelihoods and threshold for anomalies
        # ---------------------------------------------------
        log_likelihoods = gmm.score_samples(X_hist_scaled)  # array of shape (n_history,)
        # Choose threshold: e.g., 5th percentile of historical log-likelihoods
        threshold = np.percentile(log_likelihoods, 5)
        # Identify historical anomalies (for illustration; usually few if threshold low)
        is_hist_anomaly = log_likelihoods < threshold
        if df_new is not None and not df_new.empty:
            X_new_scaled = scaler.transform(df_new)
            loglik_new = gmm.score_samples(X_new_scaled)
            is_new_anomaly = loglik_new < threshold
            # We don't color by cluster; just mark anomalies vs normal
            X_new_pca_flag = True
            X_new_pca = None  # placeholder; will compute after PCA fit below
        else:
            is_new_anomaly = np.array([])
            loglik_new = np.array([])
            X_new_pca_flag = False

        label2 = "Anomaly" if is_new_anomaly.any() else "Normal"
        st.markdown(f"**Behavior Anomaly State:** {label2}")
        if(label1 == "Fraud" or label2 == "Anomaly"):
            # st.warning("⚠️ Fraud detected! Agent will fetch records for further investigation.")
            with st.spinner("🤖 Agent fetching records..."):
                # simulate delay or real fetch
                # time.sleep(2)
                os.environ["GOOGLE_API_KEY"] = ""
                model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
                with open('./Agent/Customer1.json', 'r') as f:
                    customer1 = f.read()

                response = model.invoke([HumanMessage(content="Analyze the following customer data and summarize key insights. The transaction is being flagged and an official will be investigating it using your report:\n\n" + customer1)])
                st.success("🤖 Agent completed the analysis.")
            st.markdown("### Agent Analysis Report")
            st.markdown("**Customer Data Summary:**")
            st.markdown(response.content)
        
