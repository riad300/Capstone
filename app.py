import streamlit as st

st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
h1 {letter-spacing:-0.5px;}
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

st.info("✅ Upload করতে বাম পাশের sidebar থেকে **🐟 Classifier** page এ যাও।")
