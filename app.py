import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")

st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and get the predicted species.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        # TODO: এখানে তোমার repo এর model inference বসবে
        st.success("Prediction: (connect model)")
        st.info("Confidence: (connect model)")
