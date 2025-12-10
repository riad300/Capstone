import streamlit as st

st.set_page_config(page_title="Versions • Fish AI", page_icon="🧾", layout="wide")

st.title("🧾 Versions / Changelog")
st.caption("Keep track of app updates like a real product.")

st.markdown("""
## v1.0.0
- Multi-page website (Home / Classifier / History / Versions)
- HuggingFace model auto-download
- Top-K predictions + confidence bar
- Save results to local history JSON

## Planned (v1.1+)
- Login system (optional)
- Better UI cards + navbar
- Export history as CSV
- Deploy API backend (FastAPI) + separate React frontend
""")
