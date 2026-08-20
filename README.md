# Grape Leaf Disease Detection using Deep Transfer Learning (VGG16)

This repository contains an end-to-end deep learning project for classifying grapevine leaf diseases using Transfer Learning with the VGG16 architecture. The application is deployed via Streamlit.

## Project Overview
Grapevine diseases pose a severe threat to agricultural yields. This project develops an automated image classification system capable of accurately diagnosing three major pathological conditions and identifying healthy leaves:
1. Grape___Black_rot
2. Grape___Esca_(Black_Measles)
3. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
4. Grape___healthy

## Dataset & Architecture
- **Dataset:** Contains 5,779 training images and 1,443 validation images categorized into 4 classes.
- **Model:** Built using VGG16 pre-trained on ImageNet as a feature extractor, coupled with a custom dense classification head. Achieved over **98.20% validation accuracy**.

## Installation & Running Locally
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

1. Install the required dependencies:
pip install streamlit tensorflow keras gdown numpy pillow torch

2. Run the Streamlit app:
streamlit run app.py

Author
Adem Şimşek


