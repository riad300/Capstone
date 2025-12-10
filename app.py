import os
import json
import time
import urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

MODEL_PATH = "fish_full_resnet50_classifier.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"

DB_PATH = "saved_predictions.json"

# ----------------------------
# STYLES (Top bar look)
# ----------------------------
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.2rem;}
.topbar {
  position: sticky; top: 0; z-index: 999;
  padding: 14px 18px;
  border-radius: 18px;
  background: rgba(17,24,39,0.75);
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px);
  margin-bottom: 18px;
}
.brand {display:flex; align-items:center; gap:12px;}
.brand h2 {margin:0; font-size: 24px; letter-spacing:-0.4px;}
.brand span {opacity:0.75; font-size: 13px;}
.card {
  padding: 18px; border-radius: 18px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
}
.small {opacity:0.78}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <h2>🐟 Fish Species AI</h2>
    <span>Professional demo • Upload → Predict → Save</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers: DB
# ----------------------------
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

# ----------------------------
# Helpers: Model
# ----------------------------
def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return
    st.info("Downloading model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded ✅")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()

@st.cache_resource
def load_artifacts():
    download_model_if_needed()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]
    state = ckpt["model_state"]

    # DataParallel prefix fix
    if isinstance(state, dict) and len(state) > 0:
        fk = next(iter(state.keys()))
        if fk.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state, strict=True)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, class_names, tfm

def predict_topk(pil_img, k=3):
    model, class_names, tfm = load_artifacts()
    x = tfm(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)
    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    return [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

# ----------------------------
# Tabs (Home bar style navigation)
# ----------------------------
tab_home, tab_predict, tab_history, tab_versions = st.tabs(["🏠 Home", "📤 Upload & Predict", "📜 History", "🧾 Versions"])

with tab_home:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What this website does")
    st.write("Upload a fish image and the model predicts the species with confidence.")
    st.write("You can also save predictions and view them in History.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.info("Go to **📤 Upload & Predict** tab to upload an image and run the model.")

with tab_predict:
    left, right = st.columns([1.05, 0.95], vertical_alignment="top")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload image")
        uploaded = st.file_uploader("Drop here or browse", type=["jpg", "jpeg", "png"])
        st.caption("Tip: Clear fish image দিলে accuracy ভালো হয়।")
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
        with colA:
            run = st.button("Predict", type="primary", use_container_width=True)
        with colB:
            save_btn = st.button("Save to History", use_container_width=True)

        if run:
            preds = predict_topk(img, k=top_k)
            best_label, best_conf = preds[0]

            if best_conf < threshold:
                st.warning("Uncertain result — try clearer image / different angle.")

            st.success(f"Prediction: {best_label}")
            st.progress(int(best_conf * 100))
            st.info(f"Confidence: {best_conf*100:.2f}%")

            st.subheader(f"Top-{top_k}")
            for label, conf in preds:
                st.write(f"- **{label}** — {conf*100:.2f}%")

            # store latest prediction in session for saving
            st.session_state["last_pred"] = {
                "ts": int(time.time()),
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
                st.success("Saved ✅ এখন **📜 History** tab এ দেখো।")
    else:
        st.markdown('<div class="small">No image uploaded yet.</div>', unsafe_allow_html=True)

with tab_history:
    data = load_db()
    if not data:
        st.info("No saved predictions yet. Upload & Predict করে Save করো।")
    else:
        st.subheader("Saved predictions")
        for item in data[:50]:
            st.markdown(f"### {item.get('best_label','-')} — {item.get('best_conf',0)*100:.2f}%")
            st.write(f"**File:** {item.get('filename','-')}")
            topk = item.get("topk", [])
            if topk:
                for t in topk:
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
- **v1.0.0**
  - Top bar style UI
  - Tabs navigation (Home / Upload & Predict / History / Versions)
  - HuggingFace model auto-download
  - Save predictions to History
""")
