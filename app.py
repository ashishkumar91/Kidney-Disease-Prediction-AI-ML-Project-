import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained VGG16 model
MODEL_PATH = "kidney_cancer_vgg16_balanced.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224  # Image size

# Function to preprocess and predict
def predict_image(img):
    # Convert PIL image to NumPy array
    img = np.array(img)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # Remove alpha channel

    # Resize to match model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Ensure image is in RGB format (PIL already loads as RGB, so no need for cvtColor)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize (same as in Colab)

    # Predict
    pred_prob = model.predict(img)[0][0]

    # Ensure threshold is correctly set
    threshold = 0.6
    confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

    # Assign label
    result = "🛑 Tumor Detected" if pred_prob > threshold else "✅ Normal Kidney"

    return result, confidence, pred_prob

# Streamlit UI
st.set_page_config(page_title="Kidney Cancer Detection (VGG16)", page_icon="🩺", layout="centered")
st.title("🔬 Renal Cancer Detection App (VGG16)")
st.write("Upload a CT scan image to check for the presence of a kidney tumor using our VGG16 model.")

# Upload Image
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    st.write("🕵️ Processing...")

    # Make prediction
    result, confidence, raw_prob = predict_image(image)

    # Display result
    st.success(f"### {result}")
    st.write(f"**Confidence: {confidence:.2%}**")
    st.write(f"**Raw Probability: {raw_prob:.4f}**")  # Debugging Output


