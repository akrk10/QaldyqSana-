import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="QaldyqSana", page_icon="♻️")

st.title("♻️ QaldyqSana")
st.caption("AI-Powered Waste Classification")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader(
"Upload waste image",
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
st.divider()
st.write("Plastic | Paper/Cardboard | Metal | Glass | Other")
