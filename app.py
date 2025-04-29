
import os
import subprocess
import sys

from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# === Config ===
model_path = "best.pt"
conf_thres = 0.01

# Load model
model = YOLO(model_path)

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

    # Predict
    results = model.predict(source=temp_path, conf=conf_thres, save=False)

    # Plot masks
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    overlay = img_cv2.copy()

    class_colors = {
        'acne': (128, 64, 255),
        'dark-circle': (0, 255, 255),
        'pore': (255, 100, 180),
        'wrinkles': (180, 0, 255)
    }

    names = model.names
    h, w = img_cv2.shape[:2]

    class_counts = {}

    if results[0].masks:
        for i, poly in enumerate(results[0].masks.xy):
            cls_id = int(results[0].boxes.cls[i])
            class_name = names[cls_id]
            color = class_colors.get(class_name.lower(), (160, 160, 160))

            points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [points], color)

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        result_img = cv2.addWeighted(img_cv2, 0.7, overlay, 0.3, 0)
    else:
        result_img = img_cv2.copy()

    # Display result image
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_column_width=True)

    # Show stats
    st.subheader("Detection Summary")
    cols = st.columns(len(class_counts))

    total = sum(class_counts.values())
    for idx, (cls, count) in enumerate(class_counts.items()):
        percent = (count / total) * 100 if total > 0 else 0

        fig, ax = plt.subplots(figsize=(2,2))
        wedges, texts = ax.pie(
            [percent, 100-percent],
            startangle=90,
            colors=[np.array(class_colors.get(cls.lower(), (160,160,160))) / 255.0, (0.9,0.9,0.9)],
            wedgeprops=dict(width=0.3)
        )
        ax.text(0, 0, f"{int(percent)}%", ha='center', va='center', fontsize=14, weight='bold')
        ax.set_aspect("equal")
        ax.set_title(cls, fontsize=10)
        
        with cols[idx]:
            st.pyplot(fig)
else:
    st.info("Please upload an image or take a photo.")
