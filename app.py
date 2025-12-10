import os
import urllib.request
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

MODEL_PATH = "fish_classifier_resnet50_ft.pth"
MODEL_URL = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_classifier_resnet50_ft.pth"


# ----------------------------
# Helpers
# ----------------------------
def download_model_if_needed():
    """Download model weights from HuggingFace once (first run)."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return  # already downloaded (basic sanity: >10MB)

    st.info("Downloading model... (first run only)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded ✅")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()


@st.cache_resource
def load_artifacts():
    """Load model + labels + transforms (cached)."""
    download_model_if_needed()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]

    # Build same architecture: ResNet50 + final FC for N classes
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Preprocess (ImageNet-style, good for pretrained ResNet fine-tune)
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

    x = tfm(pil_img).unsqueeze(0)  # (1,3,224,224)

    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    results = [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]
    return results


# ----------------------------
# UI
# ----------------------------
st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and get predicted species (Top-3).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        k = st.slider("Top-K", min_value=1, max_value=5, value=3)
    with col2:
        threshold = st.slider("Uncertainty threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

    if st.button("Predict"):
        try:
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

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")
