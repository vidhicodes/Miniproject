import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load images or use emojis for the icons
icons = {
    "Cardboard": "🟫",   # You can replace this with the path to an image for cardboard
    "Compost": "🍂",     # Use Leaf emoji for compost
    "Glass": "🟦",       # Replace with glass icon image
    "Metal": "🔩",       # Use Metal emoji
    "Paper": "📄",       # Paper emoji
    "Plastic": "♻️",     # Recycling emoji for plastic
    "Trash": "🗑️",      # Trash emoji
}

# Set the page layout
st.set_page_config(page_title="Waste Classification", layout="centered")

# Title of the app
st.title("Waste Classification Model")

# Description
st.write("Upload an image of waste and classify it into one of the categories below.")

# Display the waste categories as icons
col1, col2, col3 = st.columns(3)

categories = [
    {"name": "Cardboard", "icon": icons["Cardboard"], "color": "#FFD700"},
    {"name": "Compost", "icon": icons["Compost"], "color": "#32CD32"},
    {"name": "Glass", "icon": icons["Glass"], "color": "#4682B4"},
    {"name": "Metal", "icon": icons["Metal"], "color": "#708090"},
    {"name": "Paper", "icon": icons["Paper"], "color": "#FFFFFF"},
    {"name": "Plastic", "icon": icons["Plastic"], "color": "#20B2AA"},
    {"name": "Trash", "icon": icons["Trash"], "color": "#A9A9A9"},
]

# Display icons in the grid format
for i, category in enumerate(categories):
    if i % 3 == 0:
        with col1:
            st.markdown(f"<div style='text-align:center;'><div style='background-color:{category['color']}; "
                        f"border-radius:50%; padding:20px;'>{category['icon']}</div>"
                        f"<p style='color:white;'>{category['name']}</p></div>", unsafe_allow_html=True)
    elif i % 3 == 1:
        with col2:
            st.markdown(f"<div style='text-align:center;'><div style='background-color:{category['color']}; "
                        f"border-radius:50%; padding:20px;'>{category['icon']}</div>"
                        f"<p style='color:white;'>{category['name']}</p></div>", unsafe_allow_html=True)
    else:
        with col3:
            st.markdown(f"<div style='text-align:center;'><div style='background-color:{category['color']}; "
                        f"border-radius:50%; padding:20px;'>{category['icon']}</div>"
                        f"<p style='color:white;'>{category['name']}</p></div>", unsafe_allow_html=True)

# Image upload section
st.write("### Upload an image for classification:")

image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define the model loading function
@st.cache_resource  # Cache the model to reuse it without reloading each time
def load_model_from_file():
    model_path = './weights/modelnew.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)  # Load your trained model
        return model
    else:
        st.error("Model weights file not found. Please check the path.")
        return None

# Load the model
model = load_model_from_file()

def preprocess_image(image: Image.Image):
    """Preprocess the image to match the input shape of the model"""
    target_size = (224, 224)  # Replace with your model's input size
    image = image.resize(target_size)  # Resize image
    img_array = keras_image.img_to_array(image)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Perform prediction if the model is loaded and an image is uploaded
if model and image_file is not None:
    # Load and display the uploaded image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(image_array)

    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get class labels (assuming gen_labels function returns a list of class names)
    def gen_labels():
        return ["Cardboard", "Compost", "Glass", "Metal", "Paper", "Plastic", "Trash"]

    labels = gen_labels()
    predicted_label = labels[predicted_class[0]]

    # Display the predicted class
    st.write(f"### Predicted Class: **{predicted_label}**")
else:
    st.write("Please upload an image for classification.")

# Footer or additional content
st.write("Use the categories above to classify your waste properly!")
