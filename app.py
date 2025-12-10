import os
import urllib.request
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

MODEL_PATH = "fish_resnet50_pretrained_ft.pth"
MODEL_URL = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_classifier_resnet50_ft.pth"
CLASSES_PATH = "classes.txt"

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    st.info("Model downloading... (first run only)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded ✅")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        st.stop()

def load_classes():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            "classes.txt পাওয়া যায়নি। repo root এ classes.txt যোগ করো (ImageFolder classes order অনুযায়ী)।"
        )
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if len(classes) < 2:
        raise ValueError("classes.txt এ class names ঠিকভাবে নেই।")
    return classes

@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    # weights load
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt["model_state"]

    # ✅ label mapping always from classes.txt (fixes wrong fish name)
    class_names = load_classes()

    # build model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state)
    model.eval()

    # ✅ better pretrained-resnet preprocessing
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
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    return [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and get the predicted species (Top-3).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        try:
            preds = predict_topk(img, k=3)
            best_label, best_conf = preds[0]

            # ✅ optional: uncertainty rule
            if best_conf < 0.70:
                st.warning("Uncertain result — try a clearer fish image / different angle.")
            st.success(f"Prediction: {best_label}")
            st.info(f"Confidence: {best_conf*100:.2f}%")

            st.subheader("Top-3")
            for label, conf in preds:
                st.write(f"- **{label}** — {conf*100:.2f}%")

            # ✅ debug toggle (helpful)
            with st.expander("Debug"):
                st.write("Image size:", img.size)
                st.json([{"label": l, "prob": p} for l, p in preds])

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")
