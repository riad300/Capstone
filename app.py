import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# --------------------------
# 1. Model load kora
# --------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 👉 Ei list ta tomar real fish classes diye replace korba
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(Tenra)", "Chitul", "Croaker(Poya)", "Hilsha", "Kajoli", "Meni", "Pabda", "Foli", "Puti", "Rita", "Rui", "Rupchada", "Silver carp", "Telapiya", "Carp", "Koi", "Kaikka", "Koral", "Shrimp"
]

# 👉 Nijer trained model er path dao
# উদাহরণ: "Pretrained model/best_model.pth" ইত্যাদি
MODEL_PATH = "Pretrained model/best_model.pth"


def load_model():
    # Jodi tumi custom model class use kore thako,
    # oi class ta ekhane import/define kore model = MyModel(...)
    # er por model.load_state_dict(torch.load(...)) use korba.

    # Simple shortest path: jodi pura model torch.save() diye save kore thako:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    return model


model = load_model()


# --------------------------
# 2. Image transform
# --------------------------

# 👉 Size / normalization tomar training er moto kore nao
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])


# --------------------------
# 3. Prediction function
# --------------------------

def predict_species(image: Image.Image):
    """
    Gradio eta ke call korbe.
    Input: PIL Image
    Output: dict -> {class_name: probability}
    """
    # Ensure 3-channel RGB
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Gradio Label er jonno class -> prob dict
    result = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }
    return result


# --------------------------
# 4. Gradio interface
# --------------------------

demo = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="pil", label="Upload a fish image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted species"),
    title="Fish Species Detection and Classification",
    description=(
        "Upload an image of a fish and the model will predict the species. "
        "This demo is based on our self-supervised learning approach "
        "using Dino + EfficientNet."
    ),
    examples=None,  # ইচ্ছা করলে কিছু example image path দিতে পারো
)

if __name__ == "__main__":
    demo.launch()

