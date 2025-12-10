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
    """
    Supports:
    - ckpt = {"model_state": ..., "class_names": ...}
    - ckpt = {"state_dict": ...}
    - ckpt = pure state_dict
    - ckpt = {"model": state_dict, "classes": ...} (best-effort)
    """
    class_names = None

    if isinstance(ckpt, dict):
        # common fields
        if "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
            class_names = list(ckpt["class_names"])

        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state = ckpt["model_state"]
            return state, class_names

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
            return state, class_names

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
            return state, class_names

        # if it looks like a state_dict already
        if any(k.startswith(("conv1.", "layer1.", "fc.", "module.")) for k in ckpt.keys()):
            return ckpt, class_names

    # fallback: assume it's a state_dict
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

    # If class_names not found, create placeholders (won't crash)
    if not class_names:
        # try infer num_classes from fc.weight shape
        n = None
        if isinstance(state, dict) and "fc.weight" in state:
            n = state["fc.weight"].shape[0]
        class_names = [f"class_{i}" for i in range(n or 1)]

    # Build model (ResNet50 classifier)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    # IMPORTANT: strict=False so it doesn't crash on small mismatches
    missing, unexpected = model.load_state_dict(state, strict=False)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    debug = {
        "num_classes": len(class_names),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": missing[:10],
        "unexpected_keys_sample": unexpected[:10],
    }
    return model, class_names, tfm, debug

def predict_topk(pil_img: Image.Image, k: int = 3):
    model, class_names, tfm, _ = load_artifacts()
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
    st.caption("Tip: low confidence হলে uncertain দেখাবে।")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    colA, colB, colC = st.columns([0.5, 0.3, 0.2])
    run = colA.button("Predict", type="primary", use_container_width=True)
    save_btn = colB.button("Save to History", use_container_width=True)
    show_debug = colC.checkbox("Debug")

    if run:
        preds = predict_topk(img, k=top_k)
        best_label, best_conf = preds[0]

        if best_conf < threshold:
            st.warning("Uncertain result — try clearer image / different angle.")

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
        st.subheader("Debug info (load_state_dict)")
        st.json(dbg)
else:
    st.caption("Upload an image to start.")
