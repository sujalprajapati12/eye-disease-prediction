import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ================================
# Load Model
# ================================
MODEL_PATH = "1.keras"
model = load_model(MODEL_PATH)

# Define Class Labels
class_names = ['Bulging_Eyes', 'cataract', 'Crossed_Eyes', 'Glaucoma', 'retina_disease', 'Uveitis']

# Streamlit UI
st.title("👁️ Eye Disease Prediction System")
st.write("Upload an eye image to detect possible disease using Deep Learning (CNN).")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🩺 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # same size used in training
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0  # normalize

    # Predict
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]) * 100)

    # Display Results
    st.success(f"**Predicted Disease:** {class_names[idx]}")
    st.info(f"**Confidence:** {conf:.2f}%")
