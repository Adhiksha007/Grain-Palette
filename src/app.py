import streamlit as st
from utils import load_model_and_classes, preprocess_image
from predict import predict_rice_type
from PIL import Image

st.set_page_config(page_title="GrainPalette - Rice Classifier", layout="centered")

# Load model and class labels
model, class_names = load_model_and_classes()

# Sidebar for app information
st.sidebar.title("ℹ️ About GrainPalette")

st.sidebar.markdown("""
**👨‍💻Author:** Your Name  
**📄Project:** GrainPalette - Rice Grain Classifier  
**🤖Pre-trained Model:** MobileNetV2    
**🧠Technologies:**
- Streamlit  
- TensorFlow / Keras  
- Pandas
- NumPy  
- Python 3.12  

**📌Description:**  
GrainPalette is a deep learning-based web application that classifies different 
types of rice grains using **transfer learning**. By leveraging powerful 
pre-trained models, the system achieves high accuracy and efficient classification 
of rice types. It allows users to upload images of rice grains and returns the 
predicted class instantly through a web interface.

**🐧GitHub:**  
[🔗 View on GitHub](https://github.com/yourusername/grainpalette)
""")


st.title("🌾 GrainPalette: Rice Type Classifier")
st.write("Upload a rice grain image and get the predicted rice variety!")

uploaded_file = st.file_uploader("📤 Upload a rice grain image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction, confidence = predict_rice_type(model, img_array, class_names)

    st.markdown("### 🧠 Prediction")
    st.success(f"Predicted Type: **{prediction}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
