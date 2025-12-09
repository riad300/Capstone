import gradio as gr
from PIL import Image

def predict(img):
    # TODO: তোমার repo এর model load + inference এখানে বসবে
    return "Prediction will appear here"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Fish Image"),
    outputs=gr.Textbox(label="Result"),
    title="🐟 Fish Species Classifier",
    description="Upload a fish image to get predicted species."
)

demo.launch()
