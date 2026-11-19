import os
import streamlit as st
import gdown
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="QaldyqSana", page_icon="♻️", layout="wide")

st.title("♻️ QaldyqSana")
st.caption("AI-Powered Waste Classification Platform")

FILE_ID = "13KWKyzBpYzWlrG9vpQvkoPYgk0gMUaWZ"
MODEL_PATH = "best_v3.pt"

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
    results = model.predict(image, conf=0.2, iou=0.45)
    result = results[0]
    
    annotated_image = result.plot()
    st.image(annotated_image, caption="AI Detection Results")
    
    categories = {
    "Plastic": 0,
    "Paper/Cardboard": 0,
    "Metal": 0,
    "Glass": 0,
    "Other": 0
}
    confidences = []
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        category = model.names[class_id]
        confidence = float(box.conf[0]) * 100
        
        if category in categories:
            categories[category] += 1
            confidences.append(confidence)
    
    st.subheader("Analysis Summary")
    
    total = sum(categories.values())
    st.metric("Total detected objects", total)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Plastic", categories["Plastic"])
        st.metric("Paper/Cardboard", categories["Paper/Cardboard"])
        st.metric("Metal", categories["Metal"])
    
    with col2:
        st.metric("Glass", categories["Glass"])
        st.metric("Other", categories["Other"])
        
    if confidences:
        st.metric(
            "Average confidence",
            f"{sum(confidences) / len(confidences):.1f}%"
        )
    else:
        st.warning("No waste detected.")
st.divider()
st.write("Plastic | Paper/Cardboard | Metal | Glass | Other")
