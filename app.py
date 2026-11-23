import os
from datetime import datetime
from io import BytesIO

import streamlit as st
import gdown
import pandas as pd
from PIL import Image
from ultralytics import YOLO
# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="QaldyqSana",
    page_icon="♻️",
    layout="wide"
)
# ==========================================
# CONSTANTS
# ==========================================
# Google Drive file "best_v3.pt" (same ID as detector.py).
# The previous ID (13KWKyzB...) pointed to a different file named "best.pt".
FILE_ID = "15FVrrLwqYpXi6W1UbEdcmESovQf95y9b"
MODEL_PATH = "best_v3.pt"
CATEGORIES = [
    "Plastic",
    "Paper/Cardboard",
    "Metal",
    "Glass",
    "Other"
]
RECYCLABLE = ["Plastic", "Paper/Cardboard", "Metal", "Glass"]


def to_category(class_name):
    """Map a model class name to one of the 5 categories (same rule as detector.py).

    Substring matching, so names such as "plastic_bottle" or "Paper-Cardboard"
    are not silently counted as Other.
    """
    name = str(class_name).lower().replace("_", " ").replace("-", " ")
    if "plastic" in name:
        return "Plastic"
    if "paper" in name or "cardboard" in name:
        return "Paper/Cardboard"
    if "metal" in name or "aluminium" in name or "aluminum" in name:
        return "Metal"
    if "glass" in name:
        return "Glass"
    return "Other"
# ==========================================
# SESSION STATE
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
# ==========================================
# LOAD YOLO MODEL
# ==========================================
@st.cache_resource
def load_model():
    
    if not os.path.exists(MODEL_PATH):
        
        downloaded = gdown.download(
            url=f"https://drive.google.com/uc?id={FILE_ID}",
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )
        
        if not downloaded or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Could not download the YOLO model."
            )
    if os.path.getsize(MODEL_PATH) < 100000:
        raise RuntimeError(
            "The model file is missing or incomplete."
        )
        
    loaded_model = YOLO(MODEL_PATH)
    
    if loaded_model is None:
        raise RuntimeError(
            "YOLO model initialization failed."
        )
        
    return loaded_model
# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("♻️ QaldyqSana")
st.sidebar.caption(
    "AI-powered waste analysis system"
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "AI Analysis",
        "Dashboard",
        "History",
        "Economics",
        "About"
    ]
)
st.sidebar.divider()
st.sidebar.info(
    "Smart waste recognition powered by "
    "Artificial Intelligence."
)
# ==========================================
# HOME
# ==========================================
if page == "Home":
    st.title("♻️ QaldyqSana")
    
    st.subheader(
        "Intelligent Waste Analysis System"
    )
    
    st.write(
        """
        QaldyqSana is an AI-powered system designed
        to recognize and classify waste using
        computer vision technology.
        
        Upload an image, detect different types
        of waste and explore the analysis results.
        """
    )
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("AI Technology", "YOLO")
    col2.metric("Waste Categories", "5")
    col3.metric("Analysis", "Automated")
    
    st.subheader("Supported Categories")
    
    category_columns = st.columns(5)
    
    for column, category in zip(
        category_columns,
        CATEGORIES
    ):
        column.markdown(f"### {category}")
        
    st.divider()
    
    st.write(
        "Navigate to **AI Analysis** to upload "
        "an image and start detecting waste."
    )
# ==========================================
# AI ANALYSIS
# ==========================================
elif page == "AI Analysis": 
    
    st.title("🔍 AI Waste Analysis")

    st.write(
        "Upload an image to identify and classify "
        "different types of waste." 
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        
        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
        
        st.subheader("Uploaded Image")
        
        st.image(
            image,
            width="stretch"
        )
        
        if st.button(
            "Analyze Image",
            type="primary"
        ):
            
            with st.spinner(
                "Loading AI model and analyzing image..."
            ):
                try:
                    model = load_model()
                    
                    results = model.predict(
                        source=image,
                        conf=0.2,
                        iou=0.45,
                        verbose=False
                    )
                    
                    result = results[0]
                    
                    counts = {
                        category: 0
                        for category in CATEGORIES
                    }
                    confidences = []
                    
                    if result.boxes is not None:
                        
                        for box in result.boxes:
                            
                            class_id = int(
                                box.cls[0].item()
                            )
                            
                            if isinstance(model.names, dict):
                                raw_name = str(
                                    model.names.get(
                                        class_id,
                                        "Other"
                                    )
                                ) 
                            else:
                                raw_name = str(
                                    model.names[class_id]
                                )
                                
                            category = to_category(raw_name)

                            confidence = float(
                                box.conf[0].item()
                            ) * 100

                            counts[category] += 1
                            confidences.append(confidence)

                    total = sum(counts.values())

                    recyclable = (
                        counts["Plastic"]
                        + counts["Paper/Cardboard"]
                        + counts["Metal"]
                        + counts["Glass"]
                    )

                    recyclable_percent = (
                        recyclable / total * 100
                        if total > 0
                        else 0
                    )

                    average_confidence = (
                        sum(confidences) / len(confidences)
                        if confidences
                        else 0
                    )

                    annotated_image = result.plot()

                    st.session_state.last_analysis = {
                        "filename": uploaded_file.name,
                        "total": total,
                        "recyclable": recyclable,
                        "recyclable_percent": recyclable_percent,
                        "average_confidence": average_confidence,
                        "counts": counts.copy()
                    }

                    st.session_state.history.append({
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                    ),
                    "Filename": uploaded_file.name,
                    "Total objects": total,
                    "Recyclable detections": recyclable,
                    "Recyclable share (%)": round(
                        recyclable_percent, 1
                    ),
                    **{category: counts[category] for category in CATEGORIES}
                })
                    st.success("Analysis completed!")

                    st.subheader("Detection Results")

                    st.image(
                        annotated_image,
                        channels="BGR",
                        width="stretch"
                    )

                    st.subheader("Waste Statistics")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Detected Objects",
                        total
                    )

                    col2.metric(
                        "Recyclable Category Detections",
                        recyclable
                    )

                    col3.metric(
                        "Recyclable Category Share",
                        f"{recyclable_percent:.1f}%"
                    )

                    st.subheader("Category Distribution")

                    st.bar_chart(counts)

                    st.subheader("Detection Details")

                    details = pd.DataFrame({
                        "Category": CATEGORIES,
                        "Detected Objects": [
                            counts[category]
                            for category in CATEGORIES
                        ]
                    })

                    st.dataframe(
                        details,
                        width="stretch",
                        hide_index=True
                    )

                    with st.expander("Model classes (from best_v3.pt)"):
                        st.write(
                            {int(k): f"{v} → {to_category(v)}" for k, v in
                             (model.names.items() if isinstance(model.names, dict) else enumerate(model.names))}
                        )

                    if confidences:
                        
                        st.metric(
                            "Average Detection Confidence",
                            f"{average_confidence:.1f}%"
                        )

                    else:
                        
                        st.info(
                            "No waste objects were detected. "
                            "Try another image."
                        )
                except Exception as error: 
                    st.error(
                        f"Analysis error: {error}"
                    )
# ==========================================
# DASHBOARD
# ==========================================
elif page == "Dashboard":
    
    st.title("📊 Dashboard")

    st.write( 
        "Overview of the waste analysis results."
    )

    history = st.session_state.history

    if not history:

        st.info(
            "No analysis data available yet. "
            "Go to AI Analysis and upload an image first."
        )
    
    else:
        
        df = pd.DataFrame(history)
        
        total_analyses = len(df)
        
        total_objects = int(
            df["Total objects"].sum()
        )
        
        total_recyclable = int(
            df["Recyclable detections"].sum()
        )
        
        recyclable_share = (
            total_recyclable / total_objects * 100
            if total_objects > 0
            else 0
        )
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Total Analyses",
            total_analyses
        )
        
        col2.metric(
            "Detected Objects",
            total_objects
        )
        
        col3.metric(
            "Recyclable Category Share",
            f"{recyclable_share:.1f}%" 
        )
        
        st.divider()
        
        st.subheader("Latest Analysis")
        
        latest = st.session_state.last_analysis
        
        if latest is not None:
            
            st.write(
                f"**File:** {latest['filename']}"
            )
            
            st.write(
                f"**Detected objects:** {latest['total']}"
            )
            
            st.write(
                f"**Average confidence:** "
                f"{latest['average_confidence']:.1f}%"
            )
            
            st.subheader(
                "Latest Category Distribution"
            )
            
            st.bar_chart(latest["counts"])
        
        st.divider()
        
        st.subheader("Analysis History Overview")
        
        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )
# ==========================================
# HISTORY
# ==========================================
elif page == "History":
    st.title("🕘 Analysis History")

    history = st.session_state.history

    if not history:
        
        st.info(
            "Your analysis history is currently empty."
        )
    else:
        df = pd.DataFrame(history)
        
        st.write(
            f"Total analyses: **{len(df)}**"
        )
        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )
        csv = df.to_csv(
            index=False
        ).encode("utf-8")
        
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="qaldysana_history.csv",
            mime="text/csv"
        )
        
        if st.button("Clear History"):
            
            st.session_state.history = []
            st.session_state.last_analysis = None
            
            st.rerun()
# ==========================================
# ECONOMICS
# ==========================================
elif page == "Economics":
    st.title("💰 Economic Estimate")

    st.write(
        "Object counts below are **measured** by the model in this session. "
        "Mass and price are **assumptions**: a photo cannot show an object's real "
        "weight or market value. Edit them to match your local recycling point."
    )

    history = st.session_state.history
    totals = {
        category: int(sum(entry.get(category, 0) for entry in history))
        for category in CATEGORIES
    }

    # Default assumptions. Prices: Kazakhstan collection points, 2024–2025 publications
    # (informburo.kz: PET 50–90 ₸/kg, aluminium ~150 ₸/kg, paper/cardboard 10–50 ₸/kg).
    # Glass price and per-item masses are unsourced estimates.
    defaults = pd.DataFrame({
        "Category": CATEGORIES,
        "Mass per object (g)": [25.0, 40.0, 15.0, 350.0, 50.0],
        "Price (₸ per kg)": [70.0, 30.0, 150.0, 5.0, 0.0],
    })
    if "economics_assumptions" not in st.session_state:
        st.session_state.economics_assumptions = defaults

    st.subheader("Assumptions (editable)")
    assumptions = st.data_editor(
        st.session_state.economics_assumptions,
        hide_index=True,
        disabled=["Category"],
        width="stretch",
        key="economics_editor",
    )
    if st.button("Reset assumptions"):
        st.session_state.economics_assumptions = defaults
        st.rerun()

    rows = []
    for _, row in assumptions.iterrows():
        category = row["Category"]
        count = totals[category]
        mass_kg = count * float(row["Mass per object (g)"]) / 1000
        value = mass_kg * float(row["Price (₸ per kg)"]) if category in RECYCLABLE else 0.0
        rows.append({
            "Category": category,
            "Detected objects (measured)": count,
            "Estimated mass (kg)": round(mass_kg, 3),
            "Estimated value (₸)": round(value, 1),
        })
    result_df = pd.DataFrame(rows)

    recyclable_mass = float(result_df.loc[result_df["Category"].isin(RECYCLABLE), "Estimated mass (kg)"].sum())
    total_value = float(result_df["Estimated value (₸)"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected objects (measured)", int(result_df["Detected objects (measured)"].sum()))
    col2.metric("Recyclable mass (estimate)", f"{recyclable_mass:.2f} kg")
    col3.metric("Recycling value (estimate)", f"{total_value:,.0f} ₸")

    if not history:
        st.info("No analyses yet. Run an analysis first; the estimate is computed from its results.")
    else:
        st.subheader("Estimate by category")
        st.dataframe(result_df, hide_index=True, width="stretch")
        st.bar_chart(result_df.set_index("Category")["Estimated value (₸)"])

    st.caption(
        "Value = detected objects × assumed mass × assumed price. "
        "Not a profit figure: collection, transport and sorting costs are not included. "
        "'Other' is treated as non-recyclable."
    )
# ==========================================
# ABOUT
# ==========================================
elif page == "About":
    st.title("ℹ️ About QaldyqSana")

    st.subheader(
        "Artificial Intelligence-Based Waste Analysis"
    )
    st.write(
        """
        QaldyqSana is a project focused on improving
        waste identification through artificial
        intelligence and computer vision.

        The system uses a YOLO object detection model
        to recognize five waste categories.
        """
    )

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "YOLO",
        "Ultralytics",
        "Computer Vision",
        "Pandas"
    ]
    
    for technology in technologies:
        st.write(f"• {technology}")
        
    st.subheader("Waste Categories")

    for category in CATEGORIES:
        st.write(f"• {category}")

    st.divider()

    st.caption(
        "QaldyqSana — Smart Waste Analysis System"
    )
