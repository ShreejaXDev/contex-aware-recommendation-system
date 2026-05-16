import os
from pathlib import Path
import time

import streamlit as st
import pandas as pd
import requests
import streamlit.components.v1 as components

# -----------------------------
# Config
# -----------------------------
# For frontend file, parents[1] resolves to project root (frontend/ -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
USER_FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
BACKEND_BASE = "http://127.0.0.1:8000"

# -----------------------------
# Page setup & styles
# -----------------------------
st.set_page_config(page_title="AI-Powered Fashion Recommendation Engine", page_icon="🤖", layout="centered")

# Dark/glassmorphism styles
st.markdown(
    """
    <style>
        :root{
            --card-bg: rgba(255,255,255,0.04);
            --card-glow: rgba(0,0,0,0.6);
            --accent-gradient: linear-gradient(90deg,#7b61ff,#23d5ab);
            --muted: #bdbdbd;
            --bg-1: #0f1720; /* dark navy */
            --bg-2: #0b1220; /* deeper */
            --panel: rgba(255,255,255,0.03);
            --glass: rgba(255,255,255,0.02);
            --text: #e6eef8;
        }
        html, body, .main { background: linear-gradient(180deg,var(--bg-1), var(--bg-2)) !important; color: var(--text) !important; }
        .stApp {
            background: radial-gradient(1200px 400px at 10% 10%, rgba(123,97,255,0.07), transparent),
                                    radial-gradient(800px 350px at 90% 80%, rgba(35,213,171,0.04), transparent);
        }
        .hero {
            text-align: center;
            padding: 44px 24px 22px 24px;
            backdrop-filter: blur(6px) saturate(130%);
            display: block;
        }
        .hero-panel{
            display:inline-block; padding:28px 32px; border-radius:14px; background: rgba(255,255,255,0.96);
            box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        }
        .title {
            font-size: 36px;
            font-weight: 700;
            color: #0b1220; /* black-ish */
            margin-bottom: 6px;
        }
        .subtitle { color: rgba(11,18,32,0.75); margin-bottom: 18px; }
        .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:16px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.03); }
        .card h4{ margin: 0 0 6px 0; color: var(--text); }
        .card .meta { color: rgba(230,238,248,0.6); font-size:13px }
        .generate { background: var(--accent-gradient); color: white; padding: 12px 18px; border-radius: 10px; border: none; box-shadow: 0 6px 18px rgba(123,97,255,0.12); }
        .warning { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); color: var(--text); }
        .stSelectbox>div, .stSelectbox select { background: rgba(245,245,245,0.9) !important; color: #0b1220 !important; }
        .block-container{ padding-left:48px; padding-right:48px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## Project Overview")
    st.markdown("**Context-Aware Neural Recommendation Engine**")
    st.markdown("---")
    st.markdown("**Technologies**")
    st.markdown("- TensorFlow Recommenders\n- FastAPI\n- Streamlit\n- Two-Tower Retrieval")
    st.markdown("---")
    st.markdown("**Model details**")
    st.markdown("Embedding dim: 32 (from saved model)\nTop-K retrieval via BruteForce index")
    st.markdown("---")
    st.markdown("Need backend running at:\nhttp://127.0.0.1:8000")

# -----------------------------
# Header / Hero
# -----------------------------
st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🤖 AI-Powered Fashion Recommendation Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A premium demo showcasing a context-aware two-tower recommender </div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Load trained user list
# -----------------------------
@st.cache_data
def load_user_ids(path: Path):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    # Accept common column names
    if "customer_id" in df.columns:
        col = "customer_id"
    elif "user_id" in df.columns:
        col = "user_id"
    else:
        # Fallback: take first column as ID
        col = df.columns[0]
    ids = df[col].astype(str).unique().tolist()
    # Provide a short sorted sample for usability
    sample = ids[:500]
    return sample

user_samples = load_user_ids(USER_FEATURES_PATH)

if not user_samples:
    st.warning("Could not load user list from data/processed/user_features.csv — ensure preprocessing has been run.")

col1, col2, _ = st.columns([2,1,1])
with col1:
    st.markdown("### Select Trained User")
    selected_user = st.selectbox("Choose a user (from trained users)", options=user_samples, index=0 if user_samples else None)
with col2:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    generate = st.button("Generate Personalized Recommendations", key="gen", help="Calls backend to generate recommendations")

# -----------------------------
# Helper: call backend
# -----------------------------

def fetch_recommendations(user_id: str, top_k: int = 12):
    """Call the recommendation endpoint directly with a longer timeout.

    Avoid a separate health check because the backend may be loading the
    model and respond slowly; a single longer request is more robust.
    """
    url = f"{BACKEND_BASE}/recommend/{user_id}?top_k={top_k}"
    try:
        resp = requests.get(url, timeout=40)
    except requests.exceptions.ConnectionError:
        return {"error": "backend_offline", "message": "Could not reach backend at 127.0.0.1:8000"}
    except requests.exceptions.Timeout:
        return {"error": "timeout", "message": "Backend request timed out (40s). Try again."}

    if resp.status_code != 200:
        try:
            # FastAPI errors often return JSON with detail
            body = resp.json()
        except Exception:
            body = resp.text
        return {"error": "api_error", "status_code": resp.status_code, "message": body}

    try:
        data = resp.json()
    except Exception:
        return {"error": "invalid_response", "message": "Backend returned non-JSON data"}

    return {"data": data}

# -----------------------------
# On click: generate and render
# -----------------------------
if generate:
    if not selected_user:
        st.error("No user selected — please choose a trained user.")
    else:
        with st.spinner("Generating AI-powered personalized recommendations..."):
            result = fetch_recommendations(selected_user)
            time.sleep(0.6)

        if "error" in result:
            code = result.get("error")
            if code == "backend_offline":
                st.error("Backend is offline. Start your FastAPI server and try again.")
            elif code == "timeout":
                st.warning("Request timed out. Try again or increase timeout.")
            elif code == "backend_unhealthy":
                st.error("Backend reported unhealthy status. Check the server logs.")
            else:
                st.error(f"API error: {result.get('message')}")
        else:
            data = result["data"]
            recs = data.get("recommendations") or data.get("results") or []
            if not recs:
                st.info("No recommendations returned for this user.")
            else:
                st.markdown("### Recommendations")
                # Render cards in a responsive grid (3 columns)
                cols = st.columns(3, gap="large")
                for i, rec in enumerate(recs):
                    col = cols[i % 3]
                    with col:
                        # Prepare fields with fallbacks
                        prod_id = rec.get("product_id") or rec.get("article_id") or rec.get("product_id") or "Unknown"
                        ptype = rec.get("product_type") or rec.get("product_type_name") or "Unknown"
                        color = rec.get("color") or rec.get("colour_group_name") or rec.get("colour") or "Unknown"
                        ggroup = rec.get("garment_group") or rec.get("garment_group_name") or "Unknown"
                        pgroup = rec.get("product_group") or rec.get("product_group_name") or "Unknown"
                        score = rec.get("score") or 0.0
                        rank = rec.get("rank") or (i + 1)
                        # Optional description fields
                        desc = rec.get("detail_desc") or rec.get("description") or ""

                        card_html = f"""
<div class='card' style='height:260px; display:flex; flex-direction:column; justify-content:space-between;'>
  <div style='display:flex; gap:12px; align-items:center;'>
    <div style='width:64px; height:64px; border-radius:10px; background:linear-gradient(180deg, rgba(123,97,255,0.12), rgba(35,213,171,0.08)); display:flex; align-items:center; justify-content:center; font-weight:700; color:#0b1220'>
      {rank}
    </div>
    <div style='flex:1'>
      <h4 style='margin:0; color: #0b1220'>{ptype}</h4>
      <div class='meta' style='color: rgba(11,18,32,0.7)'>Color: {color} • Garment: {ggroup}</div>
    </div>
    <div style='font-weight:700; color:#fff; background:linear-gradient(90deg,#7b61ff,#23d5ab); padding:6px 10px; border-radius:10px;'>
      {score:.3f}
    </div>
  </div>
  <div style='margin-top:10px; color: rgba(11,18,32,0.9)'>
    <div style='font-size:13px;'>Group: {pgroup}</div>
    <div style='height:8px'></div>
    <div style='font-size:13px; color: rgba(11,18,32,0.9);'><strong>ID:</strong> {prod_id}</div>
    <div style='height:6px'></div>
    <div style='font-size:12px; color: rgba(11,18,32,0.75);'>{desc}</div>
  </div>
</div>
"""
                        components.html(card_html, height=280)

# Footer / credits
st.markdown("---")
st.markdown("Built with  — Two-Tower Retrieval · FastAPI · Streamlit")
