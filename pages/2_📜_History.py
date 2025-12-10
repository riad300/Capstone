import os, json
import streamlit as st

DB_PATH = "saved_predictions.json"

st.title("📜 History")

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

data = load_db()
if not data:
    st.info("No saved predictions yet.")
else:
    for item in data[:50]:
        st.markdown(f"### {item.get('best_label','-')} — {item.get('best_conf',0)*100:.2f}%")
        st.write(f"File: {item.get('filename','-')} | Mode: {item.get('mode','REAL_AI')}")
        st.divider()

if st.button("Delete history (local)"):
    try:
        os.remove(DB_PATH)
        st.success("History deleted ✅")
    except Exception as e:
        st.error(f"Could not delete: {e}")
