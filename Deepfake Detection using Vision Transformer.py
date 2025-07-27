import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms

# Set up preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Safe model loading
@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    state_dict = torch.load("deepfake_vit_model.pth", map_location='cpu')  # weights-only file
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Title and uploader
st.title("🔍 Deepfake Detection using Vision Transformer")
model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    label = "Fake" if pred == 0 else "Real"
    st.markdown(f"### Prediction: **{label}** ({confidence:.2%} confidence)")
