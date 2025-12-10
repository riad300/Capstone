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

st.title("🐟 Fish Species AI (Real Model)")
st.caption("Upload → Predict → Save")

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

uploaded = st.file_uploader("Upload fish image (JPG/PNG)", type=["jpg", "jpeg", "png"])
top_k = st.slider("Top-K", 1, 5, 3)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_column_width=True)

    if st.button("Predict", type="primary"):
        preds = predict_topk(img, k=top_k)
        best_label, best_conf = preds[0]
        st.success(f"Prediction: {best_label}")
        st.info(f"Confidence: {best_conf*100:.2f}%")
        st.write("Top-K:")
        for l, p in preds:
            st.write(f"- {l} — {p*100:.2f}%")

        st.session_state["last_pred"] = {
            "ts": int(time.time()),
            "filename": uploaded.name,
            "best_label": best_label,
            "best_conf": best_conf,
            "topk": [{"label": l, "prob": p} for l, p in preds],
        }

    if st.button("Save to History"):
        rec = st.session_state.get("last_pred")
        if not rec:
            st.warning("Predict first, then Save.")
        else:
            save_record(rec)
            st.success("Saved ✅")

st.divider()
st.subheader("History")
data = load_db()
if not data:
    st.caption("No saved predictions yet.")
else:
    for item in data[:20]:
        st.write(f"**{item['best_label']}** — {item['best_conf']*100:.2f}%  ({item['filename']})")
