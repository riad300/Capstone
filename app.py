import streamlit as st

st.set_page_config(page_title="Fish AI", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 28px; border-radius: 20px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);}
.kpi {padding: 18px; border-radius: 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);}
.small {opacity: 0.8}
</style>
""", unsafe_allow_html=True)

st.title("🐟 Fish Species AI")
st.caption("Professional demo web app • Image classification • Streamlit multi-page")

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.subheader("What this website does")
st.write(
    "Upload a fish image and the model predicts the species with confidence. "
    "You can also save results and view history."
)
st.markdown("**Navigate using the left sidebar:**")
st.write("- 🐟 Classifier (Upload & Predict)\n- 📜 History (Saved predictions)\n- 🧾 Versions (Changelog)")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Model", "ResNet50 Fine-tuned")
    st.markdown('<div class="small">Hosted weights on HuggingFace, auto-downloaded on first run.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Top-K", "Configurable")
    st.markdown('<div class="small">See Top-1 to Top-5 predictions with probabilities.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("Save Results", "Enabled")
    st.markdown('<div class="small">Store predictions locally (history page).</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.info("✅ Open **🐟 Classifier** from sidebar to start.")
