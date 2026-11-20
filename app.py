import os
from datetime import datetime

import streamlit as st
import gdown
from PIL import Image

from ultralytics import YOLO
st.set_page_config(
    page_title="QaldyqSana",
    page_icon="♻️",
    layout="wide"
)
FILE_ID = "13KWKyzBpYzWlrG9vpQvkoPYgk0gMUaWZ"
MODEL_PATH = "best_v3.pt"
CATEGORIES = [
    "Plastic",
    "Paper/Cardboard",
    "Metal",
    "Glass",
    "Other"
]
# ---------- DESIGN ----------

st.markdown("""
<style>
.stApp {
    background: #f5f8f5;
}
section[data-testid="stSidebar"] {
    background: #e8f1e9;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #176b45;
}
.subtitle {
    color: #63776a;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2ebe3;
}
</style>
""", unsafe_allow_html=True)
# ---------- MODEL ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        downloaded = gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )
        
        if not downloaded or not os.path.exists(MODEL_PATH):
            raise RuntimeError("Failed to download YOLO model.")
            
    if os.path.getsize(MODEL_PATH) < 100000:
        raise RuntimeError("The model file is missing or incomplete.")
        
    loaded_model = YOLO(MODEL_PATH)
    
    if loaded_model is None:
        raise RuntimeError("YOLO model initialization failed.")
    
    return loaded_model

try:
    model = load_model()
    st.write("YOLO class name:", model.names)
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()
# ---------- SESSION DATA ----------
if "history" not in st.session_state:
    st.session_state.history = []
# ---------- SIDEBAR ----------
st.sidebar.markdown("# ♻️ QaldyqSana")
st.sidebar.caption("AI Waste Classification")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "AI Analysis",
        "Dashboard",
        "History",
        "About"
    ]
)
# ---------- HOME ----------
if page == "Home":
    st.markdown(
        '<div class="main-title">QaldyqSana ♻️</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="subtitle">Smart waste classification powered by Artificial Intelligence</div>',
        unsafe_allow_html=True
    )
    st.divider()
    
    st.subheader("Making waste sorting smarter")
    
    st.write(
        "QaldyqSana is an intelligent waste analysis system "
        "that uses computer vision to identify different types "
        "of waste and support more efficient recycling."
    )
    col1, col2, col3 = st.columns(3)
    
    col1.metric("AI technology", "YOLO11n")
    col2.metric("Waste categories", "5")
    col3.metric("Analysis", "Automated")
    
    st.divider()
    
    st.subheader("Start analyzing")
    
    if st.button("Open AI Analysis", use_container_width=True):
        st.session_state.page = "AI Analysis"
        st.rerun()
# ---------- AI ANALYSIS ----------
elif page == "AI Analysis":
    
    st.title("🔍 AI Waste Analysis")
    
    st.write(
        "Upload an image and let the AI model identify "
        "the waste objects."
    )
    
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded is not None:
        
        image = Image.open(uploaded).convert("RGB")
        
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )
        if st.button("Analyze Image", type="primary"):
            with st.spinner("AI is analyzing your image..."):
                
                results = model.predict(
                    image,
                    conf=0.10, 
                    iou=0.45,
                    imgsz=640
                    verbose=False
                )
                
                result = results[0]
                
                counts = {category: 0 for category in CATEGORIES}
                confidences = []
                
                aliases = {
                    "plastic": "Plastic",
                    "paper": "Paper/Cardboard",
                    "paper/cardboard": "Paper/Cardboard",
                    "cardboard": "Paper/Cardboard",
                    "metal": "Metal",
                    "glass": "Glass", 
                    "other": "Other"
                }
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    raw_name = str(model.names[class_id])

                    st.write({
                        "class_id": class_id,
                        "raw_name": raw_name,
                        "confidence": float(box.conf[0])
                    })
                   
                    category = aliases.get(
                        raw_name.lower(),
                        "Other"
                    )
                    confidence = float(box.conf[0]) * 100
                    
                    counts[category] += 1
                    confidences.append(confidence)
                
                total = sum(counts.values())
                
                annotated = result.plot()

                st.success("Analysis completed!")
                
                st.subheader("Detection Results")
                
                st.image(
                    annotated,
                    caption="AI detection",
                    channels="BGR",
                    use_container_width=True
                )
                
                st.subheader("Analysis Summary")
                
                st.metric("Total detected objects", total)
                
                cols = st.columns(5)
                
                for col, category in zip(cols, CATEGORIES):
                    col.metric(category, counts[category])
                    
                if confidences:
                    average = sum(confidences) / len(confidences)
                    st.metric(
                        "Average confidence",
                        f"{average:.1f}%"
                    )
                
                else:
                    average = 0
                    st.info("No waste objects detected.")
                    
                st.subheader("Category Distribution")
                st.bar_chart(counts)

                st.session_state.history.append({
                    "Date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                    "Total": total,
                    **counts,
                    "Confidence": f"{average:.1f}%"
                })
# ---------- DASHBOARD ----------
elif page == "Dashboard":
    st.title("📊 Dashboard")
    
    history = st.session_state.history
    
    if not history:
        st.info("No analysis data yet. Analyze an image first.")
    else:
        total_analyses = len(history)
        total_objects = sum(item["Total"] for item in history)
        
        col1, col2 = st.columns(2)

        col1.metric("Total analyses", total_analyses)
        col2.metric("Objects detected", total_objects)

        totals = {
            category: sum(item[category] for item in history)
            for category in CATEGORIES
        }

        st.subheader("Detected waste by category")
        st.bar_chart(totals)
# ---------- HISTORY ----------
elif page == "History":
    st.title("🕘 Analysis History")
    
    history = st.session_state.history
    
    if not history:
        st.info("Your analysis history is empty.")
    else:
        st.dataframe(
            history,
            use_container_width=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
# ---------- ABOUT ----------
elif page == "About":
    st.title("🌱 About QaldyqSana")
    
    st.write(
        "QaldyqSana is a project focused on applying "
        "artificial intelligence to automated waste classification."
    )
    
    st.subheader("Technologies")
    
    st.markdown("""
    - **YOLO11n** — object detection
    - Computer Vision — image analysis
    - Python — application logic
    - Streamlit — interactive interface
    """)
    
    st.subheader("Waste categories")
    
    for category in CATEGORIES:
        st.write(f"♻️ {category}")
        
    st.divider()
    
    st.caption("QaldyqSana — Smarter sorting for a cleaner future.")
