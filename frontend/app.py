"""
Streamlit frontend for Plant Disease Detection
"""

from pathlib import Path

import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and let the AI predict the disease."
)

# This creates the button: like Choose File
# uploaded_file = st.file_uploader(
#     "Choose a leaf image",
#     type=["jpg", "jpeg", "png"]
# )

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key=0

uploaded_file = st.file_uploader(
    "Choose a lead image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Shows the uploaded leaf before prediction.
    st.image(
        image,
        caption = "Uploaded Image",
        use_container_width = True,
    )

    # Nothing happens until the user clicks it.
    if st.button("Predict"):
        
        with st.spinner("Predicting..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

        # This is the most important line, Instead of calling, Predictor.predict(), directly, the frontend talks to FastAPI.
            response = requests.post(
                API_URL,
                files=files,
                timeout=60,
            )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed")

            st.subheader("Result")

            st.write(
                f"**Disease:** {result['disease']}"
            )

            st.write(
                f"**Confidence:** {result['confidence']}%"
            )

        else:
            st.error(
                "Prediction failed."
            )

            st.write(response.text)

    if st.button("🔄 Upload Another Image"):
        st.session_state.uploader_key += 1
        st.rerun()