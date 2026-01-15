import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI-Driven Biodiversity Monitoring",
    layout="wide",
    page_icon="🌿"
)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌍 Project Information")
st.sidebar.markdown("""
**Project:**  
AI-Driven Biodiversity Monitoring and Species Classification  

**SDG Alignment:**  
SDG 15 – Life on Land  

**Purpose:**  
Support biodiversity awareness using AI-based species identification.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🐾 Supported Species")
st.sidebar.markdown("""
- Dog  
- Cat  
- Horse  
- Elephant  
- Butterfly  
- Spider  
- Chicken  
- Cow  
- Sheep  
- Squirrel  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Responsible AI Notice**")
st.sidebar.caption(
    "Predictions are for decision support only and may require human verification."
)

# ==============================
# MAIN TITLE
# ==============================
st.markdown(
    """
    <h1 style='text-align: center;'>🌿 AI-Driven Biodiversity Monitoring</h1>
    <h4 style='text-align: center; color: gray;'>
    Species Classification System (SDG 15 – Life on Land)
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write(
    "Upload an animal image to identify the species using a **Deep Learning model**. "
    "This application demonstrates how AI can support biodiversity conservation."
)

# ==============================
# MODEL LOADING
# ==============================
MODEL_PATH = "model/biodiversity_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# CLASS LABELS
# ==============================
CLASS_NAMES = [
    "Dog (Cane)",
    "Chicken (Gallina)",
    "Sheep (Pecora)",
    "Spider (Ragno)",
    "Squirrel (Scoiattolo)",
    "Horse (Cavallo)",
    "Cat (Gatto)",
    "Cow (Mucca)",
    "Elephant (Elefante)",
    "Butterfly (Farfalla)"
]

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("### 📤 Upload an Animal Image")
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]

    # Top-3 predictions
    top_indices = prediction.argsort()[-3:][::-1]

    with col2:
        st.markdown("### 🧠 Prediction Results")

        for i, idx in enumerate(top_indices):
            st.write(
                f"**{i+1}. {CLASS_NAMES[idx]}** — {prediction[idx]*100:.2f}%"
            )

        confidence = prediction[top_indices[0]]

        # Confidence interpretation
        st.markdown("---")
        if confidence >= 0.85:
            st.success("🔹 High confidence prediction (Reliable)")
        elif confidence >= 0.6:
            st.warning("🔸 Medium confidence (Verification recommended)")
        else:
            st.error("🔻 Low confidence (Manual review required)")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption(
    "This application demonstrates **responsible AI usage** for biodiversity conservation. "
    "Predictions should be used as decision-support tools, aligned with **UN SDG 15 – Life on Land**."
)

