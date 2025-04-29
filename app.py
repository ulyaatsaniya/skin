# app.py

import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import tempfile

# === Config ===
MODEL_PATH = "best.onnx"
CONF_THRESHOLD = 0.01
CLASSES = ['acne', 'dark-circle', 'pore', 'wrinkles']
CLASS_COLORS = {
    'acne': (180, 128, 255),
    'dark-circle': (255, 255, 0),
    'pore': (255, 100, 180),
    'wrinkles': (180, 0, 255)
}

# === Load ONNX Model ===
session = ort.InferenceSession(MODEL_PATH)

# === Streamlit UI ===
st.set_page_config(page_title="Skin Analyzer 🧑‍⚕️", layout="centered")
st.title("Skin Analyzer 🧑‍⚕️")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
open_camera = st.button("📷 Take a Photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif open_camera:
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for ONNX model
    img = np.array(image.resize((640, 640)))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: img})

    # Dummy parsing (replace with your real ONNX output parser)
    pred_classes = np.random.choice(CLASSES, 10)

    # Visualization (mask simulation)
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    for cls in pred_classes:
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        x0, y0 = np.random.randint(0, image.width-100), np.random.randint(0, image.height-100)
        x1, y1 = x0 + np.random.randint(30, 80), y0 + np.random.randint(30, 80)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

    st.image(annotated_img, caption="Predicted Regions", use_column_width=True)

    # Summary
    st.subheader("Detection Summary")
    class_counts = {cls: (pred_classes == cls).sum() for cls in CLASSES}
    total = sum(class_counts.values())

    cols = st.columns(len(class_counts))
    for idx, (cls, count) in enumerate(class_counts.items()):
        percent = (count / total) * 100 if total > 0 else 0
        with cols[idx]:
            st.metric(label=cls.capitalize(), value=f"{int(percent)}%")
else:
    st.info("Please upload or capture an image.")
