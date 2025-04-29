import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

# === Config ===
model_path = "your_model.onnx"  # Ganti nama modelmu
conf_thres = 0.01

# === Load model ===
@st.cache_resource
def load_model():
    return ort.InferenceSession(model_path)

model = load_model()

# === Preprocessing helper ===
def preprocess(image):
    img = image.resize((640, 640))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

# === Postprocessing helper ===
def postprocess(outputs, conf_thres):
    preds = outputs[0][0]  # (N, 6): x1, y1, x2, y2, conf, class
    detections = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf > conf_thres:
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
    return detections

# === Streamlit App ===
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
    input_tensor = preprocess(image)
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
    detections = postprocess(outputs, conf_thres)

    # Draw detections
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    class_colors = {
        0: (128, 64, 255),
        1: (0, 255, 255),
        2: (255, 100, 180),
        3: (180, 0, 255)
    }
    class_names = {
        0: "acne",
        1: "dark-circle",
        2: "pore",
        3: "wrinkles"
    }

    class_counts = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        color = class_colors.get(cls_id, (160, 160, 160))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        label = f"{class_names.get(cls_id, 'unknown')} {conf:.2f}"
        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((x1, max(y1-15, 0)), label, fill=color, font=font)

        class_counts[class_names.get(cls_id, 'unknown')] = class_counts.get(class_names.get(cls_id, 'unknown'), 0) + 1

    # Display result
    st.image(overlay, caption="Analyzed Image", use_column_width=True)

    # Detection Summary
    if class_counts:
        st.subheader("Detection Summary")
        cols = st.columns(len(class_counts))

        total = sum(class_counts.values())
        for idx, (cls, count) in enumerate(class_counts.items()):
            percent = (count / total) * 100 if total > 0 else 0

            fig, ax = plt.subplots(figsize=(2,2))
            wedges, texts = ax.pie(
                [percent, 100-percent],
                startangle=90,
                colors=[np.array(class_colors.get(idx, (160,160,160))) / 255.0, (0.9,0.9,0.9)],
                wedgeprops=dict(width=0.3)
            )
            ax.text(0, 0, f"{int(percent)}%", ha='center', va='center', fontsize=14, weight='bold')
            ax.set_aspect("equal")
            ax.set_title(cls, fontsize=10)

            with cols[idx]:
                st.pyplot(fig)
else:
    st.info("Please upload an image or take a photo.")