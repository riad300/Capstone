import os
import json
import datetime as dt
import streamlit as st

st.set_page_config(page_title="History • Fish AI", page_icon="📜", layout="wide")

DB_PATH = "saved_predictions.json"

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

st.title("📜 Prediction History")
st.caption("Saved results from the Classifier page.")

data = load_db()
if not data:
    st.info("No saved predictions yet. Go to 🐟 Classifier and click **Save this result**.")
    st.stop()

# show latest first
for item in data[:50]:
    ts = item.get("ts", 0)
    when = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"### {item.get('best_label','-')}  —  {item.get('best_conf',0)*100:.2f}%")
    st.write(f"**File:** {item.get('filename','-')}  |  **Time:** {when}")
    topk = item.get("topk", [])
    if topk:
        st.write("Top-K:")
        for t in topk:
            st.write(f"- {t['label']} — {t['prob']*100:.2f}%")
    st.divider()

if st.button("Delete history (local)"):
    try:
        os.remove(DB_PATH)
        st.success("History deleted ✅")
    except Exception as e:
        st.error(f"Could not delete: {e}")
