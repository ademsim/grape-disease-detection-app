import os
# Keras için PyTorch backend ayarı (Çakışmaları önler)
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras
import numpy as np
from PIL import Image

# Google Drive file ID
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

# Modeli yükle
with st.spinner("Preparing the model, please wait..."):
    model = load_my_model()

# Sınıf İsimleri (Eğitim sırasıyla birebir aynı olmalı)
class_labels = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy'
]

# --- ARAYÜZ VE ANALİZ KISMI ---

st.title("🍇 Grape Leaf Disease Detection")
st.write("Please upload a grape leaf photo to detect diseases.")

# Dosya yükleme bileşeni
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi ekranda göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Model is analyzing the image..."):
            try:
                # RGBA veya farklı formatları standart RGB'ye çevir
                image = image.convert("RGB")
                
                # Modelin beklediği giriş boyutu (170x170)
                img = image.resize((170, 170)) 
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizasyon
                
                # Batch boyutu ekle (1, 170, 170, 3)
                img_array = np.expand_dims(img_array, axis=0) 
                
                # Tahmin yapma (4 sınıf için softmax çıktıları döner)
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                predicted_label = class_labels[predicted_class_idx]
                
                st.success("Analysis Complete!")
                
                # Sonucu ekrana yazdır
                if predicted_label == 'Grape___healthy':
                    st.success(f"Result: **{predicted_label}** - Confidence: %{confidence:.2f}")
                else:
                    st.error(f"Result: **{predicted_label}** - Confidence: %{confidence:.2f}")
                
                # İsteğe bağlı tüm olasılıkları görmek için
                with st.expander("See all class probabilities"):
                    for idx, label in enumerate(class_labels):
                        st.write(f"{label}: %{predictions[0][idx]*100:.2f}")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
