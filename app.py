import os
import urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

# ✅ তোমার HF link (full weights ফাইল এখানে)
MODEL_PATH = "fish_classifier_resnet50_ft.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_classifier_resnet50_ft.pth"


def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return
    st.info("Downloading model... (first run only)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded ✅")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()


def _unwrap_state_dict(ckpt):
    """
    Handle different checkpoint formats:
    - {"model_state": ...}
    - {"state_dict": ...}
    - {"model": ...}
    - raw state_dict
    """
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes ckpt itself is already a state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _remove_module_prefix(state):
    if isinstance(state, dict) and len(state) > 0:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # labels
    if isinstance(ckpt, dict) and "class_names" in ckpt:
        class_names = ckpt["class_names"]
    else:
        raise ValueError("Checkpoint এ 'class_names' নাই। training save করার সময় class_names রাখতে হবে।")

    # state_dict unwrap + clean
    state = _unwrap_state_dict(ckpt)
    state = _remove_module_prefix(state)

    # build resnet50
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    # ✅ strict load first (best)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        # If fails, give a clear diagnosis (this is the case you hit: missing conv1, bn1, layer1...)
        missing_msg = str(e)

        st.error("❌ Full model weights load হচ্ছে না।")
        st.write(
            "কারণ: তোমার `.pth` ফাইলে সম্ভবত **শুধু head/partial weights** আছে, "
            "full ResNet50 backbone (conv1, bn1, layer1...) নাই। তাই app দিয়ে fix করা যাবে না।"
        )
        st.write("Load error (short):", missing_msg[:1200])

        # helpful debug
        if isinstance(state, dict):
            st.write("Checkpoint total keys:", len(state))
            st.write("Checkpoint key sample:", list(state.keys())[:30])

        st.info(
            "✅ Solution: training notebook থেকে FULL model save করো:\n"
            "torch.save({'model_state': model.state_dict(), 'class_names': class_names}, 'full_model.pth')\n"
            "তারপর ওই full_model.pth HF এ আপলোড করে MODEL_URL replace করো।"
        )
        st.stop()

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


# ---------------- UI ----------------
st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and get predicted species (Top-3).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    k = st.slider("Top-K", 1, 5, 3)
    threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)

    if st.button("Predict"):
        preds = predict_topk(img, k=k)
        best_label, best_conf = preds[0]

        if best_conf < threshold:
            st.warning("Uncertain result — try a clearer fish image / different angle.")

        st.success(f"Prediction: {best_label}")
        st.info(f"Confidence: {best_conf*100:.2f}%")

        st.subheader(f"Top-{k}")
        for label, conf in preds:
            st.write(f"- **{label}** — {conf*100:.2f}%")

        with st.expander("Debug (optional)"):
            st.write("Image size:", img.size)
            st.json([{"label": l, "prob": p} for l, p in preds])
else:
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")
