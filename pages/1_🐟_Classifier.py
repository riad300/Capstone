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
st.caption("Upload → Predict → Save")

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
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return
    st.info("Downloading model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded ✅")

def _unwrap_checkpoint(ckpt):
    class_names = None
    if isinstance(ckpt, dict):
        if "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
            class_names = list(ckpt["class_names"])

        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"], class_names

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], class_names

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"], class_names

        # if it looks like state_dict
        if any(k.startswith(("conv1.", "layer1.", "fc.", "module.")) for k in ckpt.keys()):
            return ckpt, class_names

    return ckpt, class_names

def _strip_module_prefix(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    first_key = next(iter(state.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state, class_names = _unwrap_checkpoint(ckpt)
    state = _strip_module_prefix(state)

    if not class_names:
        n = None
        if isinstance(state, dict) and "fc.weight" in state:
            n = state["fc.weight"].shape[0]
        class_names = [f"class_{i}" for i in range(n or 1)]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    # ✅ FIX-1: training mismatch এ crop issue কমাতে Resize((224,224))
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),   # ✅ CenterCrop বাদ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    debug = {
        "num_classes": len(class_names),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": missing[:10],
        "unexpected_keys_sample": unexpected[:10],
        "class_names_sample": class_names[:20],
    }
    return model, class_names, tfm, debug

def predict_topk(pil_img: Image.Image, k: int = 3, temperature: float = 2.0):
    model, class_names, tfm, _ = load_artifacts()

    x = tfm(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]

        # ✅ FIX-2: temperature softmax (confidence over 99% কমাতে সাহায্য করে)
        probs = torch.softmax(logits / temperature, dim=0)

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
    temperature = st.slider("Softmax temperature (reduce overconfidence)", 1.0, 5.0, 2.0, 0.1)
    show_debug = st.checkbox("Debug")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    colA, colB = st.columns([0.6, 0.4])
    run = colA.button("Predict", type="primary", use_container_width=True)
    save_btn = colB.button("Save to History", use_container_width=True)

    if run:
        preds = predict_topk(img, k=top_k, temperature=temperature)
        best_label, best_conf = preds[0]
        second_conf = preds[1][1] if len(preds) > 1 else 0.0

        # ✅ FIX-3: uncertain rule (wrong হলে reject)
        if best_conf < threshold or (best_conf - second_conf) < 0.10:
            st.warning("Uncertain prediction — please upload a clearer fish image (full fish visible).")

        st.success(f"Prediction: {best_label}")
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
            "temperature": temperature,
        }

    if save_btn:
        rec = st.session_state.get("last_pred")
        if not rec:
            st.warning("Predict first, then Save.")
        else:
            save_record(rec)
            st.success("Saved ✅ Go to History page.")

    if show_debug:
        _, _, _, dbg = load_artifacts()
        st.subheader("Debug info")
        st.json(dbg)
else:
    st.caption("Upload an image to start.")
