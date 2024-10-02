import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import preprocess, model_arc, gen_labels  # Importing from utils.py

# Define the path to your model weights
model_weights_path = './weights/modelnew.h5'

@st.cache_resource  # Use cache_resource for loading the model
def load_model():
    model = model_arc()  # Ensure this function returns the correct model architecture
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)  # Load the pre-trained weights
    else:
        st.error("Model weights file not found. Please check the path.")
    return model

# Load the model once and reuse it
model = load_model()

# Streamlit app layout
st.title("Waste Classification Model")
st.write("Upload an image of waste for classification.")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and preprocess the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess(image)

    # Make prediction
    prediction = model.predict(image_array)

    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get class labels
    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    st.write(f"Predicted Class: {predicted_label}")  # Display the predicted class label
