import os
import json
import time
import urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Classifier • Fish AI", page_icon="🐟", layout="wide")

# --------- Model config ----------
MODEL_PATH = "fish_full_resnet50_classifier.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"

DB_PATH = "saved_predictions.json"

# --------- Styling ----------
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 1.2rem;}
.card {padding: 18px; border-radius: 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);}
.small {opacity: 0.8}
</style>
""", unsafe_allow_html=True)

def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return
    with st.sidebar:
        st.info("Downloading model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        with st.sidebar:
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

    # DataParallel fix
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

# --------- Sidebar ----------
with st.sidebar:
    st.header("📤 Upload")
    uploaded = st.file_uploader("Fish image", type=["jpg", "jpeg", "png"])
    st.divider()
    st.header("⚙️ Settings")
    k = st.slider("Top-K", 1, 5, 3)
    threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")

# --------- Main UI ----------
st.title("🐟 Fish Classifier")
st.caption("Upload from sidebar → Predict → Save to History")

if not uploaded:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("How to use")
    st.write("1) Left sidebar থেকে ছবি upload করো")
    st.write("2) Predict চাপো")
    st.write("3) চাইলে Save করে 📜 History page এ দেখো")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

img = Image.open(uploaded).convert("RGB")

c1, c2 = st.columns([1.1, 0.9], vertical_alignment="top")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    if st.button("Predict", type="primary", use_container_width=True):
        try:
            preds = predict_topk(img, k=k)
            best_label, best_conf = preds[0]

            if best_conf < threshold:
                st.warning("Uncertain result — try clearer image / different angle.")

            st.success(best_label)
            st.progress(int(best_conf * 100))
            st.write(f"**Confidence:** {best_conf*100:.2f}%")

            st.write(f"**Top-{k}:**")
            for label, conf in preds:
                st.write(f"- **{label}** — {conf*100:.2f}%")

            st.divider()
            if st.button("Save result to History", use_container_width=True):
                rec = {
                    "ts": int(time.time()),
                    "filename": uploaded.name,
                    "best_label": best_label,
                    "best_conf": best_conf,
                    "topk": [{"label": l, "prob": p} for l, p in preds],
                }
                save_record(rec)
                st.success("Saved ✅ Open 📜 History page.")
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.write('<span class="small">Press Predict to run inference.</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
