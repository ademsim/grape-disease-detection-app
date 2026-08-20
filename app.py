import os
# Set Keras backend to PyTorch to avoid conflicts
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras
import numpy as np
from PIL import Image

# Google Drive file ID for your trained model
FILE_ID = '1Clk2DGtaJlX9R-bllG0g42qgv-Is8eRP'
MODEL_PATH = 'grape_disease_vgg16_model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("Downloading model from Google Drive, please wait..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return keras.models.load_model(MODEL_PATH)

# Load the model
with st.spinner("Preparing the model, please wait..."):
    model = load_my_model()

# Class labels (Must match the exact order used during training)
class_labels = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy'
]

# --- USER INTERFACE & PREDICTION LOGIC ---

st.title("🍇 Grape Leaf Disease Detection")
st.write("Please upload a grape leaf photo to detect diseases.")

# File uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Model is analyzing the image..."):
            try:
                # Convert image to RGB format (handles RGBA / transparency channels)
                image = image.convert("RGB")
                
                # FIXED: Resize image to match model's expected input shape (224x224)
                img = image.resize((224, 224)) 
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalization
                
                # Add batch dimension to match expected shape: (1, 224, 224, 3)
                img_array = np.expand_dims(img_array, axis=0) 
                
                # Perform prediction
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                predicted_label = class_labels[predicted_class_idx]
                
                st.success("Analysis Complete!")
                
                # Display the result
                if predicted_label == 'Grape___healthy':
                    st.success(f"Result: **{predicted_label}** - Confidence: {confidence:.2f}%")
                else:
                    st.error(f"Result: **{predicted_label}** - Confidence: {confidence:.2f}%")
                
                # Expandable view for all class probabilities
                with st.expander("See all class probabilities"):
                    for idx, label in enumerate(class_labels):
                        st.write(f"{label}: {predictions[0][idx]*100:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
