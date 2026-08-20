import streamlit as st
import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# 1. Model Yükleme
@st.cache_resource
def load_prediction_model():
    # .keras veya .h5 modelinizi doğrudan Keras 3 ile yüklüyoruz
    model = load_model('grape_disease_detection_model.keras')
    return model

model = load_prediction_model()

# Sınıf İsimleri
class_labels = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy'
]

st.title("🍇 Grape Leaf Disease Detection")
st.write("This application detects diseases in grape leaves using a CNN model.")

uploaded_file = st.file_uploader("Choose a leaf image (.jpg, .png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    image = image.resize((170, 170))
    img_array = np.array(image) / 255.0
    
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button('Predict Disease'):
        with st.spinner('Model is analyzing...'):
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            predicted_label = class_labels[predicted_class_idx]
            
        st.success(f"Result: **{predicted_label}**")
        st.info(f"Confidence: **%{confidence:.2f}**")
