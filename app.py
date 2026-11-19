import os
import streamlit as st
import gdown
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="QaldyqSana", page_icon="♻️", layout="wide")

st.title("♻️ QaldyqSana")
st.caption("AI-Powered Waste Classification Platform")

FILE_ID = "13KWKyzBpYzWlrG9vpQvkoPYgk0gMUaWZ"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
       gdown.download(
           f"https://drive.google.com/uc?id={FILE_ID}",
           MODEL_PATH,
           quiet=False
)
    return YOLO(MODEL_PATH)

model = load_model()
st.success("AI model loaded successfully!")

uploaded = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    if st.button("Analyze waste"):
        results = model.predict(image, conf=0.15, iou=0.45)
        result = results[0]

    if len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100
            st.success(f"Category: {model.names[class_id]}")
            st.write(f"Confidence: {confidence:.1f}%")
    else:
        st.warning("No waste detected.")
st.divider()
st.write("Plastic | Paper/Cardboard | Metal | Glass | Other")
