import os, json, time, hashlib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

DB_PATH = "saved_predictions.json"

# ---------- UI styles ----------
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.2rem;}
.topbar {position: sticky; top: 0; z-index: 999; padding: 14px 18px; border-radius: 18px;
  background: rgba(17,24,39,0.80); border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px); margin-bottom: 18px;}
.brand {display:flex; align-items:center; gap:12px;}
.brand h2 {margin:0; font-size: 24px; letter-spacing:-0.4px;}
.brand span {opacity:0.75; font-size: 13px;}
.card {padding: 18px; border-radius: 18px; background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);}
.small {opacity:0.78}
.badge {display:inline-block; padding: 4px 10px; border-radius: 999px;
  background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.35); font-size: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <h2>🐟 Fish Species AI</h2>
    <span>Upload → Predict → Save (Presentation-safe demo)</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- DB helpers ----------
def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_record(record):
    data = load_db()
    data.insert(0, record)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- Demo classes (replace later with real model classes) ----------
CLASS_NAMES = [
    "Hilsha (Ilish)", "Rui", "Katla", "Pangash", "Tilapia",
    "Silver Carp", "Mrigel", "Bata", "Koi", "Shing"
]

def demo_predict(image_bytes: bytes, top_k: int = 3):
    """
    Deterministic demo prediction (no AI). Same image -> same output.
    """
    h = hashlib.sha256(image_bytes).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.default_rng(seed)

    logits = rng.normal(size=len(CLASS_NAMES))
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    idx = np.argsort(-probs)[:top_k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx]

# ---------- Tabs ----------
tab_home, tab_predict, tab_history, tab_versions = st.tabs(
    ["🏠 Home", "📤 Upload & Predict", "📜 History", "🧾 Versions"]
)

with tab_home:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What this website does")
    st.write("This is a professional demo web app for Fish Species classification.")
    st.write("For presentation reliability, it runs in **Demo mode** (no heavy AI dependencies).")
    st.markdown('<span class="badge">Demo Mode: Stable</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.info("Go to **📤 Upload & Predict** tab to upload an image and run prediction flow.")

with tab_predict:
    left, right = st.columns([1.05, 0.95], vertical_alignment="top")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload image")
        uploaded = st.file_uploader("Drop here or browse (JPG/PNG)", type=["jpg", "jpeg", "png"])
        st.caption("Tip: Clear fish image দিলে demo output consistent থাকবে.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Settings")
        top_k = st.slider("Top-K", 1, 5, 3)
        threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        colA, colB = st.columns([0.6, 0.4])
        run = colA.button("Predict", type="primary", use_container_width=True)
        save_btn = colB.button("Save to History", use_container_width=True)

        if run:
            image_bytes = uploaded.getvalue()
            preds = demo_predict(image_bytes, top_k=top_k)
            best_label, best_conf = preds[0]

            if best_conf < threshold:
                st.warning("Uncertain result — (Demo mode) try another image.")

            st.success(f"Prediction: {best_label}")
            st.progress(int(best_conf * 100))
            st.info(f"Confidence: {best_conf*100:.2f}%")

            st.subheader(f"Top-{top_k}")
            for label, conf in preds:
                st.write(f"- **{label}** — {conf*100:.2f}%")

            st.session_state["last_pred"] = {
                "ts": int(time.time()),
                "mode": "DEMO",
                "filename": uploaded.name,
                "best_label": best_label,
                "best_conf": best_conf,
                "topk": [{"label": l, "prob": p} for l, p in preds],
            }

        if save_btn:
            rec = st.session_state.get("last_pred")
            if not rec:
                st.warning("আগে Predict চালাও, তারপর Save করো।")
            else:
                save_record(rec)
                st.success("Saved ✅ Now check 📜 History tab.")
    else:
        st.caption("No image uploaded yet.")

with tab_history:
    data = load_db()
    if not data:
        st.info("No saved predictions yet.")
    else:
        st.subheader("Saved predictions")
        for item in data[:50]:
            mode = item.get("mode", "DEMO")
            st.markdown(f"### {item.get('best_label','-')} — {item.get('best_conf',0)*100:.2f}%  ({mode})")
            st.write(f"**File:** {item.get('filename','-')}")
            for t in item.get("topk", []):
                st.write(f"- {t['label']} — {t['prob']*100:.2f}%")
            st.divider()

        if st.button("Delete history (local)"):
            try:
                os.remove(DB_PATH)
                st.success("History deleted ✅")
            except Exception as e:
                st.error(f"Could not delete: {e}")

with tab_versions:
    st.subheader("Versions / Changelog")
    st.markdown("""
- **v1.0 (Presentation-safe)**  
  - Top bar + Tabs navigation  
  - Upload → Predict → Save → History  
  - Demo-mode prediction (no heavy dependencies)

- **v1.1 (After presentation)**  
  - Enable real AI inference (PyTorch + your trained weights)
""")
