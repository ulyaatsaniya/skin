import streamlit as st
import plotly.graph_objects as go
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

# === Config ===
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.01
INPUT_SIZE = (640, 640)  # expected input size

CLASS_NAMES = ['acne', 'dark-circle', 'pore', 'wrinkles']
CLASS_COLORS = {
    'acne': '#FF69B4',
    'dark-circle': '#00CED1',
    'pore': '#FFD700',
    'wrinkles': '#8A2BE2'
}

# === Load ONNX model ===
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

model = load_model()

# === Streamlit UI ===
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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # === Preprocess Image ===
    img = image.resize(INPUT_SIZE)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)

    # === Inference ===
    ort_inputs = {model.get_inputs()[0].name: img_np}
    ort_outs = model.run(None, ort_inputs)

    # === Postprocess (Fake for demo) ===
    # Real model should output masks or detections
    # For now let's simulate dummy counts
    counts = {cls: np.random.randint(1, 20) for cls in CLASS_NAMES}

    st.subheader("Detection Summary")

    cols = st.columns(len(counts))
    total = sum(counts.values())

    for idx, (cls, count) in enumerate(counts.items()):
        percent = (count / total) * 100 if total > 0 else 0

        fig = go.Figure(data=[
            go.Pie(
                values=[percent, 100 - percent],
                hole=0.7,
                marker_colors=[CLASS_COLORS.get(cls, "#CCCCCC"), "#EEEEEE"],
                textinfo='none'
            )
        ])
        fig.update_layout(
            showlegend=False,
            annotations=[dict(text=f"{int(percent)}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            width=200
        )
        with cols[idx]:
            st.plotly_chart(fig)
            st.caption(cls.capitalize())
else:
    st.info("Please upload an image or take a photo.")