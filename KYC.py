# app.py

import streamlit as st

st.set_page_config(page_title="KYC Verification", layout="centered")

# ---- UI: Title ----
st.markdown(
    "<h1 style='text-align: center; color: #004080; margin-bottom: 0.5rem;'>KYC Verification</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---- Step 1: Upload Aadhar ----
st.markdown("### 1. Upload your Aadhaar")
aadhar_file = st.file_uploader(
    label="Browse files to upload your Aadhaar (PDF/JPG/PNG)",
    type=["pdf", "jpg", "jpeg", "png"]
)

# ---- Step 2: Capture Live Face ----
st.markdown("### 2. Live Face Check")
st.write("Use your webcam to capture a live selfie for liveness detection.")
img_data = st.camera_input("Click to open your camera and take a photo")

# ---- Step 3: Start KYC Button ----
st.markdown("### 3. Start KYC")
if st.button("Start KYC"):
    # Require both Aadhaar upload and webcam capture
    if aadhar_file is None:
        st.error("❌ Please upload your Aadhaar before starting KYC.")
    elif img_data is None:
        st.error("❌ Please capture your live photo for liveness check.")
    else:
        # Here you would call your real backend liveness & ID verification logic.
        # For this demo, we assume that if both inputs exist, KYC passes.
        st.info("🔍 Verifying documents and liveness...")
        # Simulate processing delay
        with st.spinner("Processing..."):
            import time
            time.sleep(2)
        # Display result
        st.success("✅ KYC Verified!")
        # Optionally show previews
        st.markdown("**Uploaded Aadhaar Preview:**")
        if aadhar_file.type.startswith("image"):
            st.image(aadhar_file, width=300)
        else:
            st.write(f"• Uploaded file: {aadhar_file.name}")
        st.markdown("**Live Capture Preview:**")
        st.image(img_data, width=300)
