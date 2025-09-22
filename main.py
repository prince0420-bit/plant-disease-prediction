import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# Load model and class indices
# -------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/predicted_model/plant_disease_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# -------------------------------
# Prediction function
# -------------------------------
def predict_image_class(model, uploaded_file, class_indices):
    # ✅ Automatically detect model input size
    expected_height, expected_width = model.input_shape[1:3]

    # Load and preprocess image
    img = load_img(uploaded_file, target_size=(expected_height, expected_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map index to class name
    class_labels = list(class_indices.keys())
    predicted_class = class_labels[predicted_class_index]

    return predicted_class, predictions

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image.resize((200, 200)), caption="Uploaded Image")

    with col2:
        if st.button("🔍 Classify"):
            predicted_class, predictions = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"✅ Prediction: **{predicted_class}**")

            # Show top-3 predictions
            st.subheader("📊 Top Predictions")
            class_labels = list(class_indices.keys())
            probs = predictions[0]

            # Sort top-3
            top_indices = probs.argsort()[-3:][::-1]
            for i in top_indices:
                st.write(f"{class_labels[i]}: {probs[i]*100:.2f}%")
