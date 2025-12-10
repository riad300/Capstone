import streamlit as st

st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
h1 {letter-spacing:-0.5px;}
.small {opacity: 0.8}
</style>
""", unsafe_allow_html=True)

st.title("🐟 Fish Species AI")
st.caption("Professional demo web app • Image classification • Streamlit multi-page")

with st.container(border=True):
    st.subheader("What this website does")
    st.write(
        "Upload a fish image and the model predicts the species with confidence. "
        "You can also save results and view history."
    )
    st.markdown("**Navigate using the left sidebar:**")
    st.write("- 🐟 Classifier (Upload & Predict)")
    st.write("- 📜 History (Saved predictions)")
    st.write("- 🧾 Versions (Changelog)")

st.write("")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.metric("Model", "ResNet50 Fine-tuned")
        st.caption("Hosted weights on HuggingFace, auto-downloaded on first run.")

with c2:
    with st.container(border=True):
        st.metric("Top-K", "Configurable")
        st.caption("See Top-1 to Top-5 predictions with probabilities.")

with c3:
    with st.container(border=True):
        st.metric("Save Results", "Enabled")
        st.caption("Store predictions locally (History page).")

st.info("✅ Open **🐟 Classifier** from the left sidebar to start.")
