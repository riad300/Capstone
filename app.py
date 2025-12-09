import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

@st.cache_resource
def load_model():
    ckpt = torch.load("fish_classifier_resnet50_ft.pth", map_location="cpu")
    class_names = ckpt["class_names"]

    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, class_names, tfm

def predict(pil_img):
    model, class_names, tfm = load_model()
    x = tfm(pil_img).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    return class_names[idx.item()], float(conf.item())

st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and get the predicted species.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        try:
            label, conf = predict(img)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf*100:.2f}%")
        except FileNotFoundError:
            st.error("Model file 'fish_classifier_resnet50_ft.pth' পাওয়া যাচ্ছে না। repo root এ upload করো.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.caption("Tip: clear fish image দিলে accuracy ভালো হয়.")
