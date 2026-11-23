import os
from datetime import datetime
from io import BytesIO

import streamlit as st
import gdown
import pandas as pd
import plotly.graph_objects as go
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
# CONSTANTS (internal identifiers — unchanged)
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
# DISPLAY LABELS (user-facing only; internal keys stay in English)
# ==========================================
KK = {
    "Plastic": "Пластик",
    "Paper/Cardboard": "Қағаз/Картон",
    "Metal": "Металл",
    "Glass": "Шыны",
    "Other": "Басқа",
}
CAT_COLOR = {
    "Plastic": "#2F6FDE",
    "Paper/Cardboard": "#D98A0B",
    "Metal": "#66778A",
    "Glass": "#0E9AB3",
    "Other": "#8350D8",
}
CAT_NOTE = {
    "Plastic": "Бөтелкелер, қаптамалар, пакеттер",
    "Paper/Cardboard": "Қағаз парақтары, қораптар, картон",
    "Metal": "Алюминий банкалар, қақпақтар",
    "Glass": "Шыны бөтелкелер мен банкалар",
    "Other": "Қайта өңдеуге жатпайтын қалдық",
}
HISTORY_COLUMNS_KK = {
    "Date": "Күні мен уақыты",
    "Filename": "Файл атауы",
    "Total objects": "Нысандар саны",
    "Recyclable detections": "Қайта өңделетін нысандар",
    "Recyclable share (%)": "Қайта өңделетін үлес (%)",
    **KK,
}

PAGE_HOME = "Басты бет"
PAGE_SCAN = "AI сканері"
PAGE_DASH = "Бақылау тақтасы"
PAGE_HIST = "Талдау тарихы"
PAGE_ECON = "Экономикалық бағалау"
PAGE_ABOUT = "Жоба туралы"
PAGES = [PAGE_HOME, PAGE_SCAN, PAGE_DASH, PAGE_HIST, PAGE_ECON, PAGE_ABOUT]

# Line icons (inline SVG, stroke = currentColor)
ICONS = {
    "recycle": '<path d="M7 19H4.8a1.8 1.8 0 0 1-1.6-2.7L5.5 12"/><path d="m11 19h8.2a1.8 1.8 0 0 0 1.6-2.7l-1.2-2"/><path d="m14 16-3 3 3 3"/><path d="M8.3 9.2 9.6 7a1.8 1.8 0 0 1 3.1 0l1.2 2.1"/><path d="m5.6 15.3-.8-4.2 4.1.9"/><path d="m18.5 9.2-4.1.9.8-4.2"/>',
    "bolt": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "Plastic": '<path d="M10 2h4v3l2 3v12a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2V8l2-3z"/><path d="M8 12h8"/>',
    "Paper/Cardboard": '<path d="M6 2h9l3 3v17H6z"/><path d="M15 2v3h3M9 10h6M9 14h6M9 18h4"/>',
    "Metal": '<ellipse cx="12" cy="5" rx="6" ry="2"/><path d="M6 5v14c0 1.1 2.7 2 6 2s6-.9 6-2V5"/><path d="M6 12c0 1.1 2.7 2 6 2s6-.9 6-2"/>',
    "Glass": '<path d="M9 2h6v4a4 4 0 0 1 2 3.5V20a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V9.5A4 4 0 0 1 9 6z"/><path d="M7 14h10"/>',
    "Other": '<path d="M4 7h16M10 11v6M14 11v6M5 7l1 13a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-13M9 7V4h6v3"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v4h16v-4"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "leaf": '<path d="M5 21c0-9 5-15 15-16-1 10-7 15-15 16z"/><path d="M5 21c3-5 6-8 10-10"/>',
}


def icon(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def page_header(title, subtitle, icon_name=None):
    # previous design: large plain title with a muted subtitle (icon_name kept for compatibility)
    html(f'''<div class="qs-pagehead"><h1>{title}</h1><p>{subtitle}</p></div>''')


def metric_card(label, value, note="", accent=False):
    return (f'<div class="qs-metric{" qs-metric-accent" if accent else ""}"><div class="qs-metric-label">{label}</div>'
            f'<div class="qs-metric-value">{value}</div>'
            f'{f"<div class=qs-metric-note>{note}</div>" if note else ""}</div>')


def metric_row(cards):
    html('<div class="qs-metrics">' + "".join(cards) + "</div>")


def category_chart(counts, height=320):
    """Horizontal bar chart of category counts (Kazakh labels, fixed category order)."""
    labels = [KK[c] for c in CATEGORIES][::-1]
    values = [counts.get(c, 0) for c in CATEGORIES][::-1]
    colors = [CAT_COLOR[c] for c in CATEGORIES][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           text=values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=30, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=14, color="#14231B"),
                      xaxis=dict(showgrid=True, gridcolor="#E3EAE2", zeroline=False, rangemode="tozero", dtick=1 if max(values + [0]) <= 10 else None),
                      yaxis=dict(showgrid=False), bargap=0.35)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def share_donut(recyclable, other, height=260):
    fig = go.Figure(go.Pie(labels=["Қайта өңделетін", "Басқа"], values=[recyclable, other], hole=0.62,
                           marker=dict(colors=["#10B26C", "#C9D3CB"]), sort=False,
                           textinfo="percent", hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                      legend=dict(orientation="h", y=-0.08), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=14, color="#14231B"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ==========================================
# STYLE
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
/* Previous QaldyqSana design: sticky white header with pill navigation, dark-green hero on Home,
   calm white cards on a light grey-green ground; forest brand, emerald accent, five fixed category colours. */
:root{
  --bg:#F2F5F3; --surface:#FFFFFF; --surface-2:#E8EFEB; --ink:#0E261D; --muted:#52665D; --line:#D9E3DD;
  --brand:#0B3D2E; --accent:#047857; --accent-bright:#10B981; --accent-soft:#D7F0E3;
  --warn:#8A5A00; --warn-soft:#FFF3D6;
  --shadow:0 1px 2px rgba(11,61,46,.05), 0 6px 20px rgba(11,61,46,.06);
}
html, body, .stApp { background:var(--bg); }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, label { font-family:'Inter', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family:'Manrope', system-ui, sans-serif !important; color:var(--ink); letter-spacing:-0.015em; }
.block-container { padding-top:0 !important; padding-bottom:0 !important; max-width:72rem; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
header[data-testid="stHeader"] { display:none; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }

/* ---------- sticky header with pill navigation ---------- */
.st-key-qs_header { position:sticky; top:0; z-index:50; background:var(--surface); border:1px solid var(--line); border-top:0;
  border-radius:0 0 1rem 1rem; margin:0 0 1.6rem; padding:.55rem 1rem; box-shadow:var(--shadow); }
.st-key-qs_header [data-testid="stHorizontalBlock"] { align-items:center; gap:.75rem; }
.qs-logo { display:flex; align-items:center; gap:.6rem; text-decoration:none; min-height:40px; }
.st-key-qs_header [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.st-key-qs_header [data-testid="stElementContainer"] { margin-bottom:0; }
.qs-logo-mark { width:38px; height:38px; border-radius:11px; background:var(--brand); color:#34D399; display:flex; align-items:center; justify-content:center; flex:none; }
.qs-logo-name { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.25rem; color:var(--ink); white-space:nowrap; }
.st-key-qs_header [role="radiogroup"] { display:flex; flex-wrap:nowrap; gap:.25rem; overflow-x:auto; scrollbar-width:none; justify-content:flex-start; }
.st-key-qs_header [role="radiogroup"]::-webkit-scrollbar { display:none; }
.st-key-qs_header [data-testid="stColumn"], .st-key-qs_header [data-testid="stElementContainer"],
.st-key-qs_header [data-testid="stRadio"], .st-key-qs_header [data-testid="stRadio"] > div { min-width:0 !important; max-width:100% !important; }
.st-key-qs_header [role="radiogroup"] { width:100%; max-width:100%; }
.st-key-qs_header [role="radiogroup"] label { position:relative; }   /* keeps Streamlit's hidden inputs inside the pill */
.st-key-qs_header { overflow:hidden; }
.st-key-qs_header [role="radiogroup"] label { margin:0 !important; padding:.45rem .8rem; border-radius:.65rem; cursor:pointer; white-space:nowrap; transition:background .15s, color .15s; }
.st-key-qs_header [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) { display:none; }  /* hide radio circle */
.st-key-qs_header [role="radiogroup"] label p { font-size:.95rem; font-weight:500; color:var(--muted); }
.st-key-qs_header [role="radiogroup"] label:hover { background:var(--surface-2); }
.st-key-qs_header [role="radiogroup"] label:has(input:checked) { background:var(--accent-soft); }
.st-key-qs_header [role="radiogroup"] label:has(input:checked) p { color:var(--accent); font-weight:600; }
@media (max-width: 900px) {
  .st-key-qs_header [data-testid="stHorizontalBlock"] { flex-direction:column; align-items:stretch; gap:.35rem; }
  .st-key-qs_header [data-testid="stColumn"] { width:100% !important; flex:1 1 100% !important; }
}

/* ---------- page header ---------- */
.qs-pagehead { margin:.4rem 0 1.4rem; }
.qs-pagehead h1 { font-size:clamp(1.75rem,3.2vw,2.25rem); font-weight:700; margin:0; padding:0; }
.qs-pagehead p { margin:.35rem 0 0; color:var(--muted); font-size:1.05rem; max-width:42rem; }

/* ---------- cards & metrics ---------- */
.qs-card { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.3rem 1.4rem; box-shadow:var(--shadow); }
.qs-metrics { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px,1fr)); gap:1rem; margin:.3rem 0 1.3rem; }
.qs-metric { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.1rem 1.25rem; box-shadow:var(--shadow); animation:qsfade .35s ease both; }
.qs-metric-label { font-size:.9rem; color:var(--muted); }
.qs-metric-value { font-family:'Manrope',sans-serif; font-weight:700; font-size:clamp(1.7rem,2.6vw,2.25rem); color:var(--ink); margin-top:.2rem; font-variant-numeric:tabular-nums; line-height:1.15; }
.qs-metric-accent .qs-metric-value { color:var(--accent); }
.qs-metric-note { font-size:.78rem; color:var(--muted); margin-top:.15rem; }
.qs-badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.75rem; font-weight:700; padding:.2rem .65rem; border-radius:999px; }
.qs-badge-real { background:var(--accent-soft); color:var(--accent); }
.qs-badge-est { background:var(--warn-soft); color:var(--warn); }
.qs-section-title { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.2rem; color:var(--ink); margin:1.4rem 0 .65rem; display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }

/* ---------- home: hero ---------- */
.st-key-qs_hero { background:var(--brand); border-radius:1.5rem; padding:clamp(1.6rem,4.5vw,3rem) clamp(1.4rem,4vw,2.6rem); margin-bottom:2.2rem; }
.st-key-qs_hero [data-testid="stHorizontalBlock"] { align-items:center; gap:clamp(1rem,3vw,2.5rem); }
.qs-kicker { font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#6EE7B7; }
.qs-hero-title { font-family:'Manrope',sans-serif; font-weight:700; color:#fff; font-size:clamp(2.1rem,4.6vw,3.35rem); line-height:1.05; margin:.9rem 0 1rem; letter-spacing:-.02em; }
.qs-hero-text { color:#D6E9DF; font-size:clamp(1rem,1.5vw,1.12rem); line-height:1.6; margin:0 0 1.4rem; max-width:34rem; }
.st-key-qs_hero .stButton > button { background:#fff; color:var(--brand); border:1px solid #fff; font-weight:700; }
.st-key-qs_hero .stButton > button:hover { background:#E8F5EE; color:var(--brand); border-color:#E8F5EE; }
.qs-hero-link { display:inline-flex; align-items:center; gap:.4rem; margin-top:.2rem; padding:.62rem 1.15rem; border-radius:.75rem; border:1px solid rgba(255,255,255,.35); color:#fff !important; text-decoration:none !important; font-weight:700; }
.qs-hero-link:hover { background:rgba(255,255,255,.08); }
.qs-illus { background:var(--surface); border-radius:1rem; padding:.9rem; box-shadow:0 18px 40px rgba(0,0,0,.25); }
.qs-illus svg { display:block; width:100%; height:auto; border-radius:.6rem; }
.qs-chips { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin-top:.8rem; }
.qs-chip { font-size:.78rem; font-weight:700; padding:.25rem .6rem; border-radius:999px; background:var(--surface-2); color:var(--ink); }
.qs-chip-on { background:var(--accent-soft); color:var(--accent); }
.qs-chip-arrow { color:var(--muted); font-size:.8rem; }
.qs-illus-note { font-size:.75rem; color:var(--muted); margin-top:.5rem; }

/* ---------- home: features, categories, steps, CTA ---------- */
.qs-h2 { font-family:'Manrope',sans-serif; font-weight:700; font-size:clamp(1.5rem,2.6vw,1.9rem); color:var(--ink); margin:0; letter-spacing:-.015em; }
.qs-h2-sub { color:var(--muted); margin:.35rem 0 1.1rem; }
.qs-features { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px,1fr)); gap:1rem; margin-bottom:2.2rem; }
.qs-feature { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.25rem; box-shadow:var(--shadow); }
.qs-feature-icon { width:40px; height:40px; border-radius:.75rem; background:var(--accent-soft); color:var(--accent); display:flex; align-items:center; justify-content:center; margin-bottom:1rem; }
.qs-feature b { font-family:'Manrope',sans-serif; font-size:1.08rem; color:var(--ink); display:block; line-height:1.3; }
.qs-feature p { margin:.45rem 0 0; color:var(--muted); font-size:.92rem; line-height:1.5; }
.qs-cats { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); gap:1rem; margin:.3rem 0 2.2rem; }
.qs-cat { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.1rem; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }
.qs-cat:hover { transform:translateY(-2px); box-shadow:0 10px 26px rgba(11,61,46,.09); }
.qs-cat-icon { width:40px; height:40px; border-radius:.75rem; display:flex; align-items:center; justify-content:center; margin-bottom:.75rem; }
.qs-cat-name { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.05rem; color:var(--ink); display:flex; align-items:center; gap:.5rem; }
.qs-dot { width:.6rem; height:.6rem; border-radius:50%; display:inline-block; flex:none; }
.qs-cat-note { font-size:.88rem; color:var(--muted); margin-top:.2rem; line-height:1.45; }
.qs-cat-tag { margin-top:.65rem; }
.qs-steps { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:1rem; margin-bottom:2.2rem; }
.qs-step { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.2rem 1.25rem; box-shadow:var(--shadow); }
.qs-step-n { width:30px; height:30px; border-radius:50%; background:var(--accent-soft); color:var(--accent); font-weight:700; display:flex; align-items:center; justify-content:center; font-size:.9rem; margin-bottom:.7rem; }
.qs-step b { font-family:'Manrope',sans-serif; color:var(--ink); font-size:1.02rem; }
.qs-step p { margin:.3rem 0 0; color:var(--muted); font-size:.92rem; line-height:1.5; }
.st-key-qs_cta { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.4rem 1.6rem; box-shadow:var(--shadow); margin-bottom:1rem; }
.st-key-qs_cta [data-testid="stHorizontalBlock"] { align-items:center; }
.qs-cta-title { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.4rem; color:var(--ink); margin:0; }
.qs-cta-text { color:var(--muted); margin:.3rem 0 0; }

/* ---------- footer ---------- */
.qs-footer { margin:3rem 0 0; padding:1.2rem 1.2rem 1.4rem; background:var(--surface); border:1px solid var(--line); border-bottom:0;
  border-radius:1rem 1rem 0 0; display:flex; align-items:center; gap:.8rem; color:var(--muted); font-size:.88rem; }
.qs-footer .qs-logo-mark { width:32px; height:32px; border-radius:9px; }
.qs-footer b { color:var(--ink); font-family:'Manrope',sans-serif; white-space:nowrap; }
@media (max-width: 640px) { .qs-footer { flex-wrap:wrap; } }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button { border-radius:.75rem; font-weight:700; padding:.62rem 1.2rem; border:1px solid var(--line); background:var(--surface); color:var(--ink); }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--surface-2); border-color:var(--line); color:var(--ink); }
.stButton > button p, .stDownloadButton > button p { font-weight:700; font-size:1rem; }
.stButton > button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover { filter:brightness(1.1); background:var(--accent); color:#fff; }

/* ---------- upload area (Kazakh text replaces Streamlit's own) ---------- */
[data-testid="stFileUploaderDropzone"] { border:2px dashed var(--line); border-radius:1rem; background:var(--surface); padding:2rem 1.4rem; transition:border-color .15s, background .15s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--accent-bright); background:var(--accent-soft); }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display:none; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before { content:"Қалдық фотосын осында сүйреп әкеліңіз"; font-family:'Manrope',sans-serif; font-weight:700; font-size:1.05rem; color:var(--ink); display:block; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::after { content:"JPG, JPEG немесе PNG · 200 МБ-қа дейін"; font-size:.88rem; color:var(--muted); display:block; margin-top:.15rem; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] { display:none; }
[data-testid="stFileUploaderDropzone"] button::after { content:"Файлды таңдау"; font-size:.95rem; font-weight:700; margin-left:.35rem; }

/* ---------- images: always whole, never cropped or stretched ---------- */
[data-testid="stImage"] img { border-radius:.75rem; max-width:100%; height:auto !important; object-fit:contain; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border:1px solid var(--line); border-radius:.75rem; overflow:hidden; background:var(--surface); }
[data-testid="stExpander"] details { border:1px solid var(--line); border-radius:.75rem; background:var(--surface); }
[data-testid="stAlert"] { border-radius:.75rem; }
@keyframes qsfade { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) { .qs-metric { animation:none; } .qs-cat { transition:none; } }
</style>
""")

# ==========================================
# SESSION STATE
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "page" not in st.session_state:
    st.session_state.page = PAGE_HOME
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
                "YOLO моделін жүктеу мүмкін болмады."
            )
    if os.path.getsize(MODEL_PATH) < 100000:
        raise RuntimeError(
            "Модель файлы жоқ немесе толық жүктелмеген."
        )

    loaded_model = YOLO(MODEL_PATH)

    if loaded_model is None:
        raise RuntimeError(
            "YOLO моделін іске қосу сәтсіз аяқталды."
        )

    return loaded_model


def go_to(page_name):
    st.session_state.page = page_name



# Hero illustration from the previous design (labelled "Иллюстрация" on the page; not a detection result)
HERO_ART = """<svg viewBox="0 0 480 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Иллюстрация: қалдық нысандары санаттарымен белгіленген">
<rect width="480" height="280" fill="#E8EFEB"/><rect y="230" width="480" height="50" fill="#DCE6E0"/>
<g font-family="Inter, sans-serif" font-size="11" font-weight="600">
<rect x="58" y="48" width="66" height="188" rx="4" fill="none" stroke="#2F6FDE" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="58" y="34" width="56" height="18" rx="3" fill="#0E261D"/><rect x="58" y="34" width="4" height="18" fill="#2F6FDE"/><text x="67" y="47" fill="#fff">Пластик</text>
<rect x="80" y="64" width="20" height="14" rx="3" fill="#2F6FDE"/><rect x="74" y="78" width="32" height="150" rx="14" fill="#9DBDF5"/><rect x="74" y="120" width="32" height="36" fill="#2F6FDE" opacity=".35"/>
<rect x="146" y="104" width="70" height="132" rx="4" fill="none" stroke="#66778A" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="146" y="90" width="52" height="18" rx="3" fill="#0E261D"/><rect x="146" y="90" width="4" height="18" fill="#66778A"/><text x="155" y="103" fill="#fff">Металл</text>
<ellipse cx="181" cy="124" rx="26" ry="7" fill="#7C8C9E"/><rect x="155" y="124" width="52" height="102" fill="#A9B5C2"/><ellipse cx="181" cy="226" rx="26" ry="7" fill="#8796A6"/>
<rect x="238" y="74" width="76" height="162" rx="4" fill="none" stroke="#0E9AB3" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="238" y="60" width="44" height="18" rx="3" fill="#0E261D"/><rect x="238" y="60" width="4" height="18" fill="#0E9AB3"/><text x="247" y="73" fill="#fff">Шыны</text>
<rect x="252" y="94" width="48" height="16" rx="3" fill="#0E9AB3"/><rect x="248" y="110" width="56" height="118" rx="10" fill="#A8E1EA"/><rect x="256" y="124" width="10" height="80" rx="5" fill="#fff" opacity=".55"/>
<rect x="338" y="132" width="96" height="104" rx="4" fill="none" stroke="#D98A0B" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="338" y="118" width="48" height="18" rx="3" fill="#0E261D"/><rect x="338" y="118" width="4" height="18" fill="#D98A0B"/><text x="347" y="131" fill="#fff">Қағаз</text>
<rect x="358" y="150" width="56" height="18" rx="3" fill="#E9A93A"/><rect x="348" y="166" width="76" height="62" rx="4" fill="#F2C46B"/><rect x="348" y="166" width="76" height="12" fill="#E9A93A"/>
</g></svg>"""

# ==========================================
# HEADER NAVIGATION (previous design: sticky top bar with pills)
# ==========================================
with st.container(key="qs_header"):
    logo_col, nav_col = st.columns([1, 4], vertical_alignment="center")
    with logo_col:
        html(f'''<div class="qs-logo"><div class="qs-logo-mark">{icon("recycle", 22)}</div>
        <span class="qs-logo-name">QaldyqSana</span></div>''')
    with nav_col:
        page = st.radio(
            "Бөлімдер",
            PAGES,
            key="page",
            horizontal=True,
            label_visibility="collapsed"
        )
# ==========================================
# HOME
# ==========================================
if page == PAGE_HOME:
    with st.container(key="qs_hero"):
        text_col, art_col = st.columns([11, 9], gap="large", vertical_alignment="center")
        with text_col:
            html('''<div class="qs-kicker">Жасанды интеллектке негізделген қалдықтарды талдау</div>
            <div class="qs-hero-title">Ақылды қалдық сұрыптау осы жерден басталады</div>
            <p class="qs-hero-text">QaldyqSana компьютерлік көру мен YOLO моделінің көмегімен фотодағы қалдықтарды тауып,
            әр нысанды Пластик, Қағаз/Картон, Металл, Шыны немесе Басқа санатына жіктейді.</p>''')
            st.button(
                "Талдауды бастау →",
                on_click=go_to,
                args=(PAGE_SCAN,),
                key="hero_start"
            )
            html('<a class="qs-hero-link" href="#how-it-works">Қалай жұмыс істейді</a>')
        with art_col:
            html(f'''<div class="qs-illus">{HERO_ART}
              <div class="qs-chips"><span class="qs-chip">Фото жүктеу</span><span class="qs-chip-arrow">→</span>
              <span class="qs-chip">Нысандарды анықтау</span><span class="qs-chip-arrow">→</span>
              <span class="qs-chip qs-chip-on">Санат бойынша сұрыптау</span></div>
              <div class="qs-illus-note">Иллюстрация</div></div>''')

    html('<div class="qs-h2">Неге жасанды интеллектпен сұрыптау керек</div><p class="qs-h2-sub">Автоматты тану күнделікті қалдық сұрыптауға не береді.</p>')
    html(f'''<div class="qs-features">
      <div class="qs-feature"><div class="qs-feature-icon">{icon("bolt", 20)}</div><b>Жылдам сұрыптау шешімдері</b>
        <p>Бір фото жеткілікті: модель кадрдағы әр нысанды өзі тауып, жақтаумен белгілейді.</p></div>
      <div class="qs-feature"><div class="qs-feature-icon">{icon("layers", 20)}</div><b>Бес нақты санат</b>
        <p>Пластик, Қағаз/Картон, Металл, Шыны және Басқа — нәтижелер сұрыптау жәшіктеріне тікелей сәйкес келеді.</p></div>
      <div class="qs-feature"><div class="qs-feature-icon">{icon("chart", 20)}</div><b>Нақты нәтижелерден статистика</b>
        <p>Әр талдау тарихқа сақталады. Бақылау тақтасы тек анықталған нысандарды санайды, алдын ала толтырылған дерек жоқ.</p></div>
      <div class="qs-feature"><div class="qs-feature-icon">{icon("scan", 20)}</div><b>Нақты YOLO моделі</b>
        <p>Әр нәтиже best_v3.pt моделінің нақты болжамынан алынады: санаты, жақтауы және сенімділігі.</p></div>
    </div>''')

    html('<div class="qs-h2">Қалдық санаттары</div><p class="qs-h2-sub">Жүйе танитын бес санат.</p>')
    cards = []
    for category in CATEGORIES:
        color = CAT_COLOR[category]
        tag = ('<span class="qs-badge qs-badge-real">Қайта өңделеді</span>' if category in RECYCLABLE
               else '<span class="qs-badge" style="background:#E8EFEB;color:#52665D">Қайта өңделмейді</span>')
        cards.append(f'''<div class="qs-cat"><div class="qs-cat-icon" style="background:{color}1A;color:{color}">{icon(category, 22)}</div>
        <div class="qs-cat-name"><span class="qs-dot" style="background:{color}"></span>{KK[category]}</div>
        <div class="qs-cat-note">{CAT_NOTE[category]}</div><div class="qs-cat-tag">{tag}</div></div>''')
    html('<div class="qs-cats">' + "".join(cards) + "</div>")

    html('<div class="qs-h2" id="how-it-works">Жүйе қалай жұмыс істейді</div><p class="qs-h2-sub">Суреттен нәтижеге дейінгі төрт қадам.</p>')
    html('''<div class="qs-steps">
      <div class="qs-step"><div class="qs-step-n">1</div><b>Сурет жүктеу</b><p>Қалдық бейнеленген JPG немесе PNG фотосуретін таңдаңыз.</p></div>
      <div class="qs-step"><div class="qs-step-n">2</div><b>Нысандарды анықтау</b><p>YOLO моделі суреттегі әр нысанды тауып, оны жақтаумен белгілейді.</p></div>
      <div class="qs-step"><div class="qs-step-n">3</div><b>Санатқа бөлу</b><p>Әр нысан бес санаттың біріне жатқызылып, сенімділік пайызы көрсетіледі.</p></div>
      <div class="qs-step"><div class="qs-step-n">4</div><b>Статистика</b><p>Нәтижелер бақылау тақтасына, тарихқа және экономикалық бағалауға қосылады.</p></div>
    </div>''')

    with st.container(key="qs_cta"):
        cta_text, cta_btn = st.columns([3, 1], vertical_alignment="center")
        with cta_text:
            html('<p class="qs-cta-title">Қалдықтың фотосы бар ма?</p><p class="qs-cta-text">Жүктеңіз де, әр нысанды, оның санатын және модельдің сенімділігін көріңіз.</p>')
        with cta_btn:
            st.button(
                "Талдауды бастау →",
                type="primary",
                on_click=go_to,
                args=(PAGE_SCAN,),
                key="cta_start",
                width="stretch"
            )
# ==========================================
# AI ANALYSIS
# ==========================================
elif page == PAGE_SCAN:

    page_header(
        "AI сканері",
        "Суретті жүктеңіз: YOLO моделі қалдық нысандарын тауып, санаттарға бөледі.",
        "scan"
    )

    uploaded_file = st.file_uploader(
        "Сурет жүктеу",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("Талдауды бастау үшін қалдық бейнеленген суретті жүктеңіз.")

    if uploaded_file is not None:

        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        html('<div class="qs-section-title">Жүктелген сурет</div>')

        preview_col, action_col = st.columns([3, 2], gap="large")
        with preview_col:
            st.image(
                image,
                width="stretch"
            )
        with action_col:
            html(f'''<div class="qs-card"><div class="qs-cat-name">{uploaded_file.name}</div>
            <div class="qs-cat-note">{image.width} × {image.height} пиксель</div>
            <p class="qs-cat-note" style="margin-top:.8rem">«Талдау» батырмасын басқанда сурет YOLO моделіне жіберіледі.
            Бірінші талдау кезінде модель жүктеліп алынады, сондықтан біраз уақыт кетуі мүмкін.</p></div>''')
            analyze = st.button(
                "Суретті талдау",
                type="primary",
                width="stretch"
            )

        if analyze:

            with st.spinner(
                "AI моделі жүктеліп, сурет талдануда..."
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
                    objects = []  # per-object rows for display only

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
                            objects.append((raw_name, category, confidence))

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
                    st.success("Талдау аяқталды. Нәтиже тарихқа сақталды.")

                    html('<div class="qs-section-title">Анықтау нәтижесі</div>')

                    st.image(
                        annotated_image,
                        channels="BGR",
                        width="stretch",
                        caption="YOLO моделі белгілеген нысандар (жақтаудағы жазулар — модельдің өз класс атаулары)"
                    )

                    metric_row([
                        metric_card("Анықталған нысандар", total, accent=True),
                        metric_card("Қайта өңделетін нысандар", recyclable, "Пластик, қағаз/картон, металл, шыны"),
                        metric_card("Қайта өңделетін үлес", f"{recyclable_percent:.1f}%"),
                        metric_card("Орташа сенімділік", f"{average_confidence:.1f}%" if confidences else "–"),
                    ])

                    if confidences:
                        html('<div class="qs-section-title">Санаттар бойынша бөліну</div>')
                        category_chart(counts)
                        if True:
                            html('<div class="qs-section-title">Әр нысан және сенімділік</div>')
                            st.dataframe(
                                pd.DataFrame({
                                    "№": range(1, len(objects) + 1),
                                    "Санат": [KK[c] for _, c, _ in objects],
                                    "Модель класы": [n for n, _, _ in objects],
                                    "Сенімділік (%)": [round(p, 1) for _, _, p in objects],
                                }),
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Сенімділік (%)": st.column_config.ProgressColumn(
                                        "Сенімділік", format="%.1f%%", min_value=0, max_value=100
                                    )
                                },
                            )

                        html('<div class="qs-section-title">Санаттар кестесі</div>')
                        st.dataframe(
                            pd.DataFrame({
                                "Санат": [KK[c] for c in CATEGORIES],
                                "Анықталған нысандар": [
                                    counts[category]
                                    for category in CATEGORIES
                                ]
                            }),
                            width="stretch",
                            hide_index=True
                        )

                    else:

                        st.info(
                            "Суретте қалдық нысаны анықталмады. "
                            "Басқа суретпен қайталап көріңіз."
                        )

                    with st.expander("Модель кластары (best_v3.pt)"):
                        st.dataframe(
                            pd.DataFrame([
                                {"ID": int(k), "Модель класы": v, "Санат": KK[to_category(v)]}
                                for k, v in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names))
                            ]),
                            width="stretch",
                            hide_index=True
                        )
                except Exception as error:
                    st.error(
                        f"Талдау кезінде қате шықты: {error}"
                    )
# ==========================================
# DASHBOARD
# ==========================================
elif page == PAGE_DASH:

    page_header(
        "Бақылау тақтасы",
        "Осы сеанстағы талдау нәтижелерінің жалпы көрінісі.",
        "chart"
    )

    history = st.session_state.history

    if not history:

        st.info(
            "Әзірге талдау деректері жоқ. "
            "Алдымен «AI сканері» бөлімінде сурет жүктеңіз."
        )
        st.button("AI сканеріне өту", type="primary", on_click=go_to, args=(PAGE_SCAN,))

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

        category_totals = {
            category: int(df[category].sum()) if category in df else 0
            for category in CATEGORIES
        }

        metric_row([
            metric_card("Талдаулар саны", total_analyses, accent=True),
            metric_card("Анықталған нысандар", total_objects),
            metric_card("Қайта өңделетін нысандар", total_recyclable),
            metric_card("Қайта өңделетін үлес", f"{recyclable_share:.1f}%"),
        ])

        chart_col, donut_col = st.columns([3, 2], gap="large")
        with chart_col:
            html('<div class="qs-section-title">Барлық талдаулар: санаттар бойынша</div>')
            category_chart(category_totals)
        with donut_col:
            html('<div class="qs-section-title">Қайта өңделетін үлес</div>')
            if total_objects > 0:
                share_donut(total_recyclable, total_objects - total_recyclable)
            else:
                st.info("Нысандар әлі анықталмады.")

        html('<div class="qs-section-title">Соңғы талдау</div>')

        latest = st.session_state.last_analysis

        if latest is not None:

            metric_row([
                metric_card("Файл", f'<span style="font-size:1rem;word-break:break-all">{latest["filename"]}</span>'),
                metric_card("Анықталған нысандар", latest["total"]),
                metric_card("Орташа сенімділік", f"{latest['average_confidence']:.1f}%"),
            ])

            category_chart(latest["counts"], height=260)

        html('<div class="qs-section-title">Талдаулар тарихына шолу</div>')

        st.dataframe(
            df.rename(columns=HISTORY_COLUMNS_KK),
            width="stretch",
            hide_index=True
        )
# ==========================================
# HISTORY
# ==========================================
elif page == PAGE_HIST:
    page_header(
        "Талдау тарихы",
        "Осы сеанста жасалған барлық талдаулар.",
        "chart"
    )

    history = st.session_state.history

    if not history:

        st.info(
            "Талдау тарихы әзірге бос."
        )
    else:
        df = pd.DataFrame(history)

        metric_row([
            metric_card("Талдаулар саны", len(df), accent=True),
            metric_card("Анықталған нысандар", int(df["Total objects"].sum())),
        ])
        display_df = df.rename(columns=HISTORY_COLUMNS_KK)
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )
        csv = display_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Тарихты CSV ретінде жүктеу",
                data=csv,
                file_name="qaldysana_history.csv",
                mime="text/csv",
                width="stretch"
            )

        with col2:
            if st.button("Тарихты тазалау", width="stretch"):

                st.session_state.history = []
                st.session_state.last_analysis = None

                st.rerun()

        st.caption("Тарих осы браузер сеансында сақталады: қосымша қайта іске қосылғанда тазаланады.")
# ==========================================
# ECONOMICS
# ==========================================
elif page == PAGE_ECON:
    page_header(
        "Экономикалық бағалау",
        "Анықталған қалдықтың болжамды массасы мен қайта өңдеу құны.",
        "chart"
    )

    html('''<div class="qs-card" style="margin-bottom:1rem">
      <span class="qs-badge qs-badge-real">Өлшенген</span> нысандар саны — модельдің осы сеанстағы нақты нәтижесі.<br>
      <span class="qs-badge qs-badge-est" style="margin-top:.45rem">Болжам</span> масса мен баға — фотодан анықталмайды.
      Оларды жергілікті қабылдау пунктінің деректеріне сай өзгертуге болады.</div>''')

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

    html('<div class="qs-section-title">Болжамдар <span class="qs-badge qs-badge-est">өзгертуге болады</span></div>')
    editor_df = st.session_state.economics_assumptions.copy()
    editor_df.insert(0, "Санат", [KK[c] for c in editor_df["Category"]])  # display column only
    edited = st.data_editor(
        editor_df,
        hide_index=True,
        disabled=["Санат", "Category"],
        column_order=["Санат", "Mass per object (g)", "Price (₸ per kg)"],
        column_config={
            "Mass per object (g)": st.column_config.NumberColumn("Бір нысан массасы (г)", min_value=0.0, format="%.1f"),
            "Price (₸ per kg)": st.column_config.NumberColumn("Бағасы (₸/кг)", min_value=0.0, format="%.1f"),
        },
        width="stretch",
        key="economics_editor",
    )
    assumptions = edited.drop(columns=["Санат"])
    if st.button("Бастапқы болжамдарға қайтару"):
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

    metric_row([
        metric_card('Анықталған нысандар <span class="qs-badge qs-badge-real">өлшенген</span>',
                    int(result_df["Detected objects (measured)"].sum()), accent=True),
        metric_card('Қайта өңделетін масса <span class="qs-badge qs-badge-est">болжам</span>', f"{recyclable_mass:.2f} кг"),
        metric_card('Қайта өңдеу құны <span class="qs-badge qs-badge-est">болжам</span>', f"{total_value:,.0f} ₸".replace(",", " ")),
    ])

    if not history:
        st.info("Әзірге талдау жоқ. Алдымен талдау жасаңыз: бағалау соның нәтижесінен есептеледі.")
    else:
        html('<div class="qs-section-title">Санаттар бойынша бағалау</div>')
        if True:
            st.dataframe(
                pd.DataFrame({
                    "Санат": [KK[c] for c in result_df["Category"]],
                    "Нысандар (өлшенген)": result_df["Detected objects (measured)"].tolist(),
                    "Масса, кг (болжам)": result_df["Estimated mass (kg)"].tolist(),
                    "Құны, ₸ (болжам)": result_df["Estimated value (₸)"].tolist(),
                }),
                hide_index=True,
                width="stretch",
            )
        if True:
            fig = go.Figure(go.Bar(
                x=[KK[c] for c in result_df["Category"]],
                y=result_df["Estimated value (₸)"],
                marker_color=[CAT_COLOR[c] for c in result_df["Category"]],
                hovertemplate="%{x}: %{y:.1f} ₸<extra></extra>",
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="₸", gridcolor="#E3EAE2"),
                              font=dict(family="Inter, sans-serif", size=13, color="#14231B"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption(
        "Құн = анықталған нысандар саны × бір нысанның болжамды массасы × 1 кг болжамды бағасы. "
        "Бұл таза пайда емес: жинау, тасымалдау және сұрыптау шығындары ескерілмеген. "
        "«Басқа» санаты қайта өңделмейді деп есептеледі."
    )
# ==========================================
# ABOUT
# ==========================================
elif page == PAGE_ABOUT:
    page_header(
        "Жоба туралы",
        "Жасанды интеллектке негізделген қалдықтарды талдау",
        "leaf"
    )

    html('''<div class="qs-card">
    <p style="margin-top:0"><b>QaldyqSana</b> — компьютерлік көру арқылы қалдықтарды тануды жақсартуға бағытталған
    ғылыми-технологиялық жоба. Жүйе фотосуреттегі қалдық нысандарын тауып, оларды бес санатқа бөледі және
    қайта өңдеуге жарамды қалдықтың үлесін көрсетеді.</p>
    <p style="margin-bottom:0">Жобаның мақсаты — қалдықтарды бастапқы сұрыптауды жеңілдету және жасанды интеллекттің
    қалдықтарды басқару мен қайта өңдеуді тиімдірек етуге қалай көмектесетінін көрсету.</p></div>''')

    html('<div class="qs-section-title">Технология</div>')
    html(f'''<div class="qs-steps">
      <div class="qs-step"><div class="qs-cat-icon" style="background:#fff;color:#0C8F57">{icon("cpu")}</div><b>Компьютерлік көру</b>
        <p>Бағдарламаға суреттердегі нысандарды тануды үйрететін жасанды интеллект саласы.</p></div>
      <div class="qs-step"><div class="qs-cat-icon" style="background:#fff;color:#0C8F57">{icon("scan")}</div><b>YOLO нысан анықтау</b>
        <p>YOLO (You Only Look Once) моделі суретті бір рет өңдеп, ондағы барлық нысандардың орнын жақтаумен белгілейді,
        класын және сенімділік деңгейін береді.</p></div>
      <div class="qs-step"><div class="qs-cat-icon" style="background:#fff;color:#0C8F57">{icon("chart")}</div><b>Талдау және статистика</b>
        <p>Анықталған нысандар санаттар бойынша есептеліп, бақылау тақтасы мен экономикалық бағалауда көрсетіледі.</p></div>
    </div>''')

    html('<div class="qs-section-title">Қалдық санаттары</div>')
    html('<div class="qs-cats">' + "".join(
        f'''<div class="qs-cat"><div class="qs-cat-icon" style="background:{CAT_COLOR[c]}1A;color:{CAT_COLOR[c]}">{icon(c, 22)}</div>
        <div class="qs-cat-name">{KK[c]}</div><div class="qs-cat-note">{CAT_NOTE[c]}</div></div>''' for c in CATEGORIES) + "</div>")

    html('<div class="qs-section-title">Экологиялық және экономикалық маңызы</div>')
    html('''<div class="qs-card"><ul style="margin:0;padding-left:1.2rem;line-height:1.7">
      <li>Қайта өңделетін материал аралас қоқысқа түссе, ластанып, полигонға кетеді. Дұрыс сұрыптау полигонға түсетін қалдықты азайтады.</li>
      <li>Пластик, қағаз, металл және шыны — қабылдау пункттерінде сатылатын шикізат. Дәл сұрыптау олардың құнын сақтайды.</li>
      <li>Автоматты тану сұрыптау орталықтарында, мектептерде және ақылды қоқыс жәшіктерінде қолданылуы мүмкін.</li>
    </ul></div>''')

    st.caption("QaldyqSana — қалдықтарды ақылды талдау жүйесі · Python, Streamlit, Ultralytics YOLO")

# ==========================================
# FOOTER (all pages)
# ==========================================
html(f'''<div class="qs-footer"><div class="qs-logo-mark">{icon("recycle", 18)}</div><b>QaldyqSana</b>
<span>Жасанды интеллектке негізделген қалдықтарды анықтау және жіктеу платформасы · YOLO (best_v3.pt), Streamlit</span></div>''')
