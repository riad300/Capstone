import os, json, time, urllib.request
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fish Species AI", page_icon="🐟", layout="wide")

MODEL_PATH = "fish_full_resnet50_classifier.pth"
MODEL_URL  = "https://huggingface.co/riad300/fish-resnet50-weights/resolve/main/fish_full_resnet50_classifier.pth"
DB_PATH = "saved_predictions.json"

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.2rem;}
.topbar {position: sticky; top: 0; z-index: 999; padding: 14px 18px; border-radius: 18px;
  background: rgba(17,24,39,0.75); border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(10px); margin-bottom: 18px;}
.brand {display:flex; align-items:center; gap:12px;}
.brand h2 {margin:0; font-size: 24px;}
.brand span {opacity:0.75; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <h2>🐟 Fish Species AI</h2>
    <span>Upload → Predict → Save</span>
  </div>
</div>
""", unsafe_allow_html=True)

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_record(record):
    data = load_db()
    data.insert(0, record)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000:
        return True, ""
    try:
        st.info("Downloading model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded ✅")
        return True, ""
    except Exception as e:
        return False, f"Model download failed: {e}"

@st.cache_resource
def try_load_model():
    """
    This function NEVER crashes the whole app.
    It returns (ok, model_bundle_or_error_message)
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        ok, msg = download_model_if_needed()
        if not ok:
            return False, msg

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        class_names = ckpt["class_names"]
        state = ckpt["model_state"]

        # DataParallel fix
        if isinstance(state, dict) and len(state) > 0:
            fk = next(iter(state.keys()))
            if fk.startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(state, strict=True)
        model.eval()

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return True, (model, class_names, tfm, torch)

    except Exception as e:
        return False, f"AI init failed: {repr(e)}"

def predict_topk(pil_img, k=3):
    ok, bundle = try_load_model()
    if not ok:
        return False, bundle

    model, class_names, tfm, torch = bundle
    x = tfm(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)

    top_probs, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    preds = [(class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]
    return True, preds

# ---------- UI ----------
tab_home, tab_predict, tab_history, tab_versions = st.tabs(
    ["🏠 Home", "📤 Upload & Predict", "📜 History", "🧾 Versions"]
)

with tab_home:
    st.subheader("Status")
    ok, bundle = try_load_model()
    if ok:
        st.success("AI is ready ✅")
    else:
        st.warning("AI not ready (app still works).")
        st.code(bundle)

with tab_predict:
    uploaded = st.file_uploader("Upload fish image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    top_k = st.slider("Top-K", 1, 5, 3)
    threshold = st.slider("Uncertainty threshold", 0.0, 1.0, 0.70, 0.01)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        colA, colB = st.columns([0.6, 0.4])
        run = colA.button("Predict", type="primary", use_container_width=True)
        save_btn = colB.button("Save to History", use_container_width=True)

        if run:
            ok, out = predict_topk(img, k=top_k)
            if not ok:
                st.error("Prediction failed:")
                st.code(out)
            else:
                preds = out
                best_label, best_conf = preds[0]

                if best_conf < threshold:
                    st.warning("Uncertain result — try clearer image.")

                st.success(f"Prediction: {best_label}")
                st.progress(int(best_conf * 100))
                st.info(f"Confidence: {best_conf*100:.2f}%")

                st.subheader(f"Top-{top_k}")
                for label, conf in preds:
                    st.write(f"- **{label}** — {conf*100:.2f}%")

                st.session_state["last_pred"] = {
                    "ts": int(time.time()),
                    "filename": uploaded.name,
                    "best_label": best_label,
                    "best_conf": best_conf,
                    "topk": [{"label": l, "prob": p} for l, p in preds],
                }

        if save_btn:
            rec = st.session_state.get("last_pred")
            if not rec:
                st.warning("আগে Predict চালাও, তারপর Save করো।")
            else:
                save_record(rec)
                st.success("Saved ✅ Check 📜 History tab.")

with tab_history:
    data = load_db()
    if not data:
        st.info("No saved predictions yet.")
    else:
        for item in data[:50]:
            st.markdown(f"### {item.get('best_label','-')} — {item.get('best_conf',0)*100:.2f}%")
            st.write(f"**File:** {item.get('filename','-')}")
            st.divider()

with tab_versions:
    st.markdown("- **v1.0** — Crash-proof UI + debug output in Home tab")
