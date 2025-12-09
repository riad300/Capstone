import os
import urllib.request
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

MODEL_PATH = "fish_resnet50_pretrained_ft.pth"
MODEL_URL = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_resnet50_pretrained_ft.pth"

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

@st.cache_resource
def load_artifacts():
    download_model_if_needed()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
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
        preds = predict_topk(img, k=3)
        best_label, best_conf = preds[0]
        st.success(f"Prediction: {best_label}")
        st.info(f"Confidence: {best_conf*100:.2f}%")

        st.subheader("Top-3")
        for label, conf in preds:
            st.write(f"- **{label}** — {conf*100:.2f}%")
else:
    st.caption("Tip: পরিষ্কার মাছের ছবি দিলে accuracy ভালো হয়।")
