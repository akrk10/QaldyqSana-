import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import os

st.set_page_config(
page_title="QaldyqSana",
page_icon="♻️",
layout="wide"
)

st.title("♻️ QaldyqSana")
st.caption("AI-Powered Waste Classification Platform")

MODEL_PATH = "best.pt"
FILE_ID = "13KWKyzBpYzWlrG9vpQvkoPYgk0gMUaWZ"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

return YOLO(MODEL_PATH)
try:
model = load_model()
st.success("AI model loaded successfully!")

uploaded = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image")

    if st.button("Analyze waste"):
        with st.spinner("AI is analyzing..."):
            results = model.predict(image)
            result = results[0]

        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0]) * 100
                category = model.names[class_id]

                st.success(f"Category: {category}")
                st.write(f"Confidence: {confidence:.1f}%")
        else:
            st.warning("No waste detected.")
except Exception as e:
st.error(f"Model loading error: {e}")

st.divider()
st.write("Plastic | Paper/Cardboard | Metal | Glass | Other")
