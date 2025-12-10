import os
import json
import time
import urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Classifier • Fish Species AI", page_icon="🐟", layout="wide")

# ---------------------------
# CONFIG (✅ update only if needed)
# ---------------------------
MODEL_PATH = "fish_full_resnet50_classifier.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"

DB_PATH = "saved_predictions.json"

# ---------------------------
# STYLE
# ---------------------------
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 1.2rem;}
.card {padding: 18px; border-radius: 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);}
.small {opacity: 0.8}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# STORAGE
# ---------------------------
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

# ---------------------------
# MODEL LOADING
# ---------------------------
def download_model_if_needed():
    # basic sanity: if already downloaded & > 10MB, skip
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return

    st.sidebar.info("Downloading model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.sidebar.success("Model downloaded ✅")
    except Exception as e:
        st.sidebar.error(f"Model download failed: {e}")
        st.stop()

@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # must exist in full checkpoint
    class_names = ckpt["class_names"]
    state = ckpt["model_state"]

    # DataParallel fix: remove "module." prefix
    if isinstance(state, dict) and len(state) > 0:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # build the same model architecture
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

    x = tfm(pil_img).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    return [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

# ---------------------------
# SIDEBAR (Upload + settings)
# ---------------------------
with st.sidebar:
    st.header("📤 Upload")
    uploaded = st.file_uploader("Fish image", type=["jpg", "jpeg", "png"])
    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K", 1, 5, 3)
    threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")

# ---------------------------
# MAIN UI
# ---------------------------
st.title("🐟 Fish Classifier")
st.caption("Left sidebar থেকে image upload করো → Predict → Save করলে History তে যাবে")

if not uploaded:
    with st.container(border=True):
        st.subheader("How to use")
        st.write("1) বাম পাশের sidebar খুলে ছবি upload করো")
        st.write("2) Predict চাপো")
        st.write("3) Save করলে 📜 History page এ চলে যাবে")
    st.stop()

img = Image.open(uploaded).convert("RGB")

c1, c2 = st.columns([1.1, 0.9], vertical_alignment="top")

with c1:
    with st.container(border=True):
        st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

with c2:
    with st.container(border=True):
        st.subheader("Prediction")
        run = st.button("Predict", type="primary", use_container_width=True)

        if run:
            try:
                preds = predict_topk(img, k=top_k)
                best_label, best_conf = preds[0]

                if best_conf < threshold:
                    st.warning("Uncertain result — try clearer image / different angle.")

                st.success(f"Prediction: {best_label}")
                st.progress(int(best_conf * 100))
                st.info(f"Confidence: {best_conf*100:.2f}%")

                st.write(f"**Top-{top_k}:**")
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
                    st.success("Saved ✅ Now open 📜 History page from sidebar menu.")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.caption("Press **Predict** to run inference.")
