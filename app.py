import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Define a function to process the uploaded image
def process_image(img):
    img = img.resize((64, 64))  # Resize the image to match the model input
    img = np.array(img)  # Convert the image to a numpy array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size
    return img

# Set page config for a better title and centered layout
st.set_page_config(page_title="Flower Recognition", page_icon="🌸", layout="centered")

# Title with emoji
st.title('🌸 Flower Recognition 🌸')

# File uploader widget with a centered label
file = st.file_uploader("Choose a flower image", type=['jpg', 'jpeg', 'png'])

# Display uploaded image if file is chosen
if file is not None:
    img = Image.open(file)
    
    # Display uploaded image with center alignment
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Process the image and make predictions
    image = process_image(img)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    # Class names
    class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

    # Show prediction in a visually appealing way with centered text
    st.markdown(f"### 🌼 **Prediction**: 🌼")
    st.markdown(f"#### The model predicts this flower as: **{class_names[predicted_class]}**")

    # Add a stylish container to make the output look cleaner
    st.markdown("""
    <style>
    .stMarkdown {
        font-size: 20px;
        text-align: center;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Center all elements and create a more modern design
st.markdown("""
    <style>
    .stApp {
        text-align: center;
        background-color: #f0f0f5;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        width: 200px;
    }
    .stTitle {
        color: #4CAF50;
    }
    .stImage img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Optional footer or additional message
st.markdown("""
    <footer style="text-align: center;">
    <p style="color: #333333; font-size: 14px;">Created with ❤️ by Senasu Demir</p>
    </footer>
""", unsafe_allow_html=True)
