# app.py

import streamlit as st
import requests

# Backend URL (change to your actual backend address)
BACKEND_URL = "http://localhost:5000/api/transaction"

st.set_page_config(page_title="SecureBank Dashboard", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; font-size: 48px;'>SecureBank</h1>",
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("## Initiate a Transaction")

if st.button("Initiate Transaction"):
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

        submitted = st.form_submit_button("Submit Transaction")
        if submitted:
            payload = {
                "to": to_account,
                "from": from_account,
                "time": transaction_time,
                "city_population": city_population,
                "latitude": latitude,
                "longitude": longitude,
                "amount": amount,
                "category": category,
                "gender": gender,
                "state": state,
                "merchant_latitude": merchant_latitude,
                "merchant_longitude": merchant_longitude
            }

            # Show a loading spinner while waiting for the backend response
            with st.spinner("Submitting transaction and fetching results..."):
                try:
                    resp = requests.post(BACKEND_URL, json=payload, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("✅ Transaction Submitted!")
                    st.json(data)  # display the backend response
                except requests.RequestException as e:
                    st.error(f"❌ Failed to submit: {e}")
