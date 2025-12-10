import os, json, time, urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

MODEL_PATH = "fish_full_resnet50_classifier.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"
DB_PATH = "saved_predictions.json"
CACHE_BUSTER = "v1"  # change to v2/v3 if you need force reload

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.2rem;}
.topbar {position: sticky; top: 0; z-index: 999; padding: 14px 18px; border-radius: 18px;
  background: rgba(17,24,39,0.75); border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px); margin-bottom: 18px;}
.brand {display:flex; align-items:center; gap:12px;}
.brand h2 {margin:0; font-size: 24px; letter-spacing:-0.4px;}
.brand span {opacity:0.75; font-size: 13px;}
.small {opacity:0.78}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <h2>🐟 Fish Species AI</h2>
    <span>Upload → Predict → Save</span>
  </div>
</div>
""", unsafe_allow_html=True)

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

def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return
    st.info("Downloading model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded ✅")

@st.cache_resource
def load_model():
    _ = CACHE_BUSTER
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
    model, class_names, tfm = load_model()
    x = tfm(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)
    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    return [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

tab_home, tab_predict, tab_history, tab_versions = st.tabs(
    ["🏠 Home", "📤 Upload & Predict", "📜 History", "🧾 Versions"]
)

with tab_home:
    st.subheader("What this website does")
    st.write("Upload a fish image and the model predicts the species with confidence.")
    st.info("Go to **📤 Upload & Predict** tab.")

with tab_predict:
    left, right = st.columns([1.05, 0.95], vertical_alignment="top")

    with left:
        st.subheader("Upload image")
        uploaded = st.file_uploader("Drop here or browse", type=["jpg", "jpeg", "png"])

    with right:
        st.subheader("Settings")
        top_k = st.slider("Top-K", 1, 5, 3)
        threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        colA, colB = st.columns([0.6, 0.4])
        run = colA.button("Predict", type="primary", use_container_width=True)
        save_btn = colB.button("Save to History", use_container_width=True)

        if run:
            preds = predict_topk(img, k=top_k)
            best_label, best_conf = preds[0]

            if best_conf < threshold:
                st.warning("Uncertain result — try clearer image.")

            st.success(f"Prediction: {best_label}")
            st.progress(int(best_conf * 100))
            st.info(f"Confidence: {best_conf*100:.2f}%")

            st.subheader(f"Top-{top_k}")
            for label, conf in preds:
                st.write(f"- **{label}** — {conf*100:.2f}%")

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
                st.success("Saved ✅ Now check 📜 History tab.")
    else:
        st.caption("No image uploaded yet.")

with tab_history:
    data = load_db()
    if not data:
        st.info("No saved predictions yet.")
    else:
        for item in data[:50]:
            st.markdown(f"### {item.get('best_label','-')} — {item.get('best_conf',0)*100:.2f}%")
            st.write(f"**File:** {item.get('filename','-')}")
            st.divider()

with tab_versions:
    st.markdown("- **v1.1 (AI)** — PyTorch inference + HF model download")
