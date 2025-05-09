import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os
from io import StringIO

# --- Custom CSS for Compact, Cooler Warm UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {
        background: linear-gradient(135deg, #f8e1d9, #f5c6aa);
        padding: 20px;
        border-radius: 15px;
        color: #6b4e31;
        font-family: 'Roboto', sans-serif;
    }
    .title {
        color: #c0392b;
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #8e6f52;
        font-size: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .sidebar-title {
        color: #c0392b;
        font-size: 24px;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(45deg, #d35400, #e67e22);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 25px;
        font-size: 16px;
        font-weight: bold;
        transition: transform 0.2s;
        margin: 5px;
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }
    .upload-box {
        border: 2px dashed #d35400;
        padding: 15px;
        border-radius: 10px;
        background: rgba(245, 198, 170, 0.3);
        text-align: center;
        color: #8e6f52;
        margin-bottom: 15px;
    }
    .upload-box:hover {
        border-color: #c0392b;
    }
    .result-text {
        color: #6b4e31;
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model from Google Drive ---
@st.cache_resource
def load_oa_model():
    model_path = 'koa_classifier_densenet121_50epochs.h5'
    if not os.path.exists(model_path):
        # Replace with your Google Drive file ID
        url = 'https://drive.google.com/uc?id=1MfI_McCpebzXZXW4_2s-owGgUCwmCYDk'
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_oa_model()

# --- Preprocessing ---
IMG_SIZE = (256, 256)

def preprocess_image(image):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=1)  # Grayscale X-ray
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = tf.repeat(img, repeats=3, axis=-1)
    img = tf.expand_dims(img, axis=0)
    return img

# --- Streamlit App ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">Knee OA Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Assess Knee Health with X-ray Analysis</p>', unsafe_allow_html=True)

# Sidebar with Compact Info and Threshold Toggle
with st.sidebar:
    st.markdown('<p class="sidebar-title">Analyzer Info</p>', unsafe_allow_html=True)
    st.write("AI-driven assessment of knee osteoarthritis using X-rays.")
    st.markdown("- **Grades**: 0 (None) to 3 (Moderate)")
    st.write("Upload an X-ray to evaluate.")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 60, help="Set minimum confidence for valid predictions")

# Upload Section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"], 
                                help="Supports .jpg, .jpeg, .png", key="uploader")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    # Save uploaded file to disk temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Zoom Slider and Image Display
    zoom_level = st.slider("Zoom Level", 1.0, 3.0, 1.0, 0.1, help="Adjust to zoom into the X-ray")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("temp_image.jpg", caption="X-ray", width=int(256 * zoom_level), clamp=True)
    
    with col2:
        col2a, col2b = st.columns([1, 1])
        with col2a:
            analyze_button = st.button("Analyze")
        with col2b:
            clear_button = st.button("Clear")

        if analyze_button:
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image("temp_image.jpg")
                prediction = model.predict(processed_img)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = prediction[0][predicted_class] * 100
                grade_labels = {0: "No OA (Grade 0)", 1: "Doubtful (Grade 1)", 
                                2: "Mild (Grade 2)", 3: "Moderate (Grade 3)"}
                predicted_grade = grade_labels[predicted_class]

            # Result Display with Threshold Check
            if confidence >= confidence_threshold:
                st.markdown(f'<p class="result-text"><b>Diagnosis:</b> {predicted_grade}</p>', unsafe_allow_html=True)
                gauge_color = "#27ae60" if confidence >= 80 else "#f1c40f" if confidence >= 60 else "#e74c3c"
                st.markdown(f"""
                    <div style="text-align: left;">
                        <p class="result-text"><b>Confidence:</b></p>
                        <div style="width: 100%; background: #f5d6c4; border-radius: 8px; height: 15px;">
                            <div style="width: {confidence}%; background: {gauge_color}; height: 100%; border-radius: 8px;"></div>
                        </div>
                        <p class="result-text" style="color: {gauge_color};">{confidence:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

                # Enhanced Output
                if predicted_class == 3:
                    st.markdown('<p class="result-text" style="color: #c0392b;"><b>Recommendation:</b> Moderate OA detected. Consult a specialist.</p>', unsafe_allow_html=True)
                elif predicted_class == 0:
                    st.markdown('<p class="result-text" style="color: #27ae60;"><b>Assessment:</b> Low risk of OA.</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="result-text"><b>Note:</b> {predicted_grade} detected.</p>', unsafe_allow_html=True)

                # Download Result
                result_text = f"Diagnosis: {predicted_grade}\nConfidence: {confidence:.2f}%\n"
                if predicted_class == 3:
                    result_text += "Recommendation: Moderate OA detected. Consult a specialist."
                elif predicted_class == 0:
                    result_text += "Assessment: Low risk of OA."
                else:
                    result_text += f"Note: {predicted_grade} detected."
                st.download_button("Download Result", data=result_text, file_name="oa_result.txt", mime="text/plain")
            else:
                st.markdown(f'<p class="result-text" style="color: #e74c3c;">Confidence ({confidence:.2f}%) below threshold ({confidence_threshold}%). Result unreliable.</p>', unsafe_allow_html=True)

        if clear_button:
            st.session_state.uploader = None  # Reset uploader
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
            st.experimental_rerun()  # Refresh app

else:
    st.markdown('<p style="color: #8e6f52; text-align: center;">Upload an X-ray to analyze.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)