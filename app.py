import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


st.set_page_config(
    page_title="AI-Driven Biodiversity Monitoring",
    layout="centered"
)


st.title("🌿 AI-Driven Biodiversity Monitoring and Species Classification System")
st.write(
    "Upload an animal image to identify the species using a Deep Learning model. "
    "This project supports **SDG 15 – Life on Land**."
)


MODEL_PATH = "model/biodiversity_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


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


uploaded_file = st.file_uploader(
    "📤 Upload an animal image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]


    st.markdown("### 🧠 Prediction Result")
    st.success(f"**Predicted Species:** {predicted_class}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")


st.markdown("---")
st.caption(
    "This application demonstrates responsible AI usage for biodiversity conservation "
    "aligned with **UN SDG 15 – Life on Land**."
)
