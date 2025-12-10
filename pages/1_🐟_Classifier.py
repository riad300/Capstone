import os, json, time, urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ====== CONFIG ======
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"
MODEL_PATH = "fish_full_resnet50_classifier.pth"
DB_PATH = "saved_predictions.json"

st.title("🐟 Classifier (Real Model)")
st.caption("Upload → Predict → Save → History")

# ====== History DB ======
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

# ====== Download model ======
def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return
    st.info("Downloading model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded ✅")

# ====== Load model (cached) ======
@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if "model_state" not in ckpt or "class_names" not in ckpt:
        raise RuntimeError("Checkpoint must contain keys: model_state, class_names")

    class_names = ckpt["class_names"]
    state = ckpt["model_state"]

    # DataParallel fix: remove "module."
    if isinstance(state, dict) and len(state) > 0:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
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

def predict_topk(pil_img: Image.Image, k: int = 3):
    model, class_names, tfm = load_artifacts()

    x = tfm(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    preds = [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]
    return preds

# ====== UI ======
colL, colR = st.columns([1.1, 0.9], vertical_alignment="top")

with colL:
    uploaded = st.file_uploader("Upload fish image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with colR:
    top_k = st.slider("Top-K", 1, 5, 3)
    threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    colA, colB = st.columns([0.6, 0.4])
    run = colA.button("Predict", type="primary", use_container_width=True)
    save_btn = colB.button("Save to History", use_container_width=True)

    if run:
        preds = predict_topk(img, k=top_k)
        best_label, best_conf = preds[0]

        # ✅ Confidence 100% দেখালে threshold দিয়ে “uncertain” দেখাবে
        if best_conf < threshold:
            st.warning("Uncertain result — try clearer image / different angle.")

        st.success(f"Prediction: {best_label}")
        st.progress(int(best_conf * 100))
        st.info(f"Confidence: {best_conf*100:.2f}%")

        st.subheader(f"Top-{top_k}")
        for label, conf in preds:
            st.write(f"- **{label}** — {conf*100:.2f}%")

        st.session_state["last_pred"] = {
            "ts": int(time.time()),
            "mode": "REAL_AI",
            "filename": uploaded.name,
            "best_label": best_label,
            "best_conf": best_conf,
            "topk": [{"label": l, "prob": p} for l, p in preds],
        }

    if save_btn:
        rec = st.session_state.get("last_pred")
        if not rec:
            st.warning("Predict first, then Save.")
        else:
            save_record(rec)
            st.success("Saved ✅ Go to History page.")
else:
    st.caption("Upload an image to start.")
