"""DeepGuard AI — Streamlit web app.

Run:
    streamlit run app.py
"""
import os

import numpy as np
import streamlit as st
import tensorflow as tf

import config
from preprocess import decode_bytes, prepare_for_model

st.set_page_config(page_title="DeepGuard AI", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(config.MODEL_PATH)


def predict(model, img: np.ndarray):
    p_real = float(model.predict(prepare_for_model(img), verbose=0)[0][0])
    if p_real >= config.THRESHOLD:
        return config.DISPLAY_NAMES["REAL"], p_real, p_real
    return config.DISPLAY_NAMES["FAKE"], 1.0 - p_real, p_real


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "DeepGuard AI uses a Convolutional Neural Network trained to tell real "
        "photographs apart from AI-generated images."
    )
    st.markdown(
        "**Pipeline**\n"
        "1. Image decoded with OpenCV\n"
        "2. Resized to 256×256\n"
        "3. Pixels normalized to [0, 1]\n"
        "4. CNN outputs P(real)"
    )
    st.caption("Predictions are probabilistic and can be wrong.")

# ---------------- Main ----------------
st.title("🛡️ DeepGuard AI")
st.subheader("AI-Generated Image Detection")

if not os.path.exists(config.MODEL_PATH):
    st.error(
        f"Trained model not found at `{config.MODEL_PATH}`.\n\n"
        "Train it first with `python train.py` (see README.md)."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded is not None:
    try:
        img = decode_bytes(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption=uploaded.name)

    with col2:
        with st.spinner("Analyzing..."):
            label, confidence, p_real = predict(model, img)

        if label == config.DISPLAY_NAMES["REAL"]:
            st.success(f"### ✅ {label}")
        else:
            st.error(f"### 🤖 {label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

        st.write("**Class probabilities**")
        st.write(f"- Real: {p_real * 100:.2f}%")
        st.write(f"- AI-Generated: {(1 - p_real) * 100:.2f}%")
else:
    st.info("Upload a JPG/PNG image to check whether it is real or AI-generated.")
