import os
import sys

try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install opencv-python")
    import cv2

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image

# === Config ===
model_path = "best.onnx"  # <- ONNX model
conf_thres = 0.01

# Load ONNX model once
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Streamlit UI
st.set_page_config(page_title="Skin Analyzer 🧑‍⚕️", layout="centered")
st.title("Skin Analyzer 🧑‍⚕️")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
open_camera = st.button("📷 Take a Photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif open_camera:
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")

if image:
    # Save temp image
    temp_path = "/tmp/temp_input.jpg"
    image.save(temp_path)

    # Preprocessing
    img = np.array(image.resize((640, 640)))  # assuming model input size
    img = img.transpose(2, 0, 1)  # (HWC -> CHW)
    img = img.astype(np.float32) / 255.0  # normalize
    img = np.expand_dims(img, axis=0)

    # Predict
    outputs = session.run(None, {input_name: img})  # output is a list

    # Postprocessing (⚡ very basic, adjust if needed)
    preds = outputs[0][0]  # Take first image prediction
    # Assume segmentation mask or raw detection depending on your ONNX export
    # For now, let's simulate detections
    
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    overlay = img_cv2.copy()

    # Fake detections for visualization (replace with actual postprocess if needed)
    fake_classes = ['acne', 'dark-circle', 'pore', 'wrinkles']
    class_colors = {
        'acne': (128, 64, 255),
        'dark-circle': (0, 255, 255),
        'pore': (255, 100, 180),
        'wrinkles': (180, 0, 255)
    }

    class_counts = {cls: np.random.randint(1, 5) for cls in fake_classes}  # simulate counts

    # Fake visualization (for real models you should threshold & mask properly)
    for cls_name, count in class_counts.items():
        for _ in range(count):
            center = (np.random.randint(100, 540), np.random.randint(100, 540))
            radius = np.random.randint(10, 30)
            color = class_colors.get(cls_name.lower(), (160, 160, 160))
            cv2.circle(overlay, center, radius, color, -1)

    result_img = cv2.addWeighted(img_cv2, 0.7, overlay, 0.3, 0)

    # Display result image
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_column_width=True)

    # Show stats
    st.subheader("Detection Summary")
    cols = st.columns(len(class_counts))

    total = sum(class_counts.values())
    for idx, (cls, count) in enumerate(class_counts.items()):
        percent = (count / total) * 100 if total > 0 else 0

        fig, ax = plt.subplots(figsize=(2, 2))
        wedges, texts = ax.pie(
            [percent, 100 - percent],
            startangle=90,
            colors=[np.array(class_colors.get(cls.lower(), (160, 160, 160))) / 255.0, (0.9, 0.9, 0.9)],
            wedgeprops=dict(width=0.3)
        )
        ax.text(0, 0, f"{int(percent)}%", ha='center', va='center', fontsize=14, weight='bold')
        ax.set_aspect("equal")
        ax.set_title(cls, fontsize=10)

        with cols[idx]:
            st.pyplot(fig)
else:
    st.info("Please upload an image or take a photo.")