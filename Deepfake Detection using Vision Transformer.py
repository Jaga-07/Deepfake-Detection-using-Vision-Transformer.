import os
import gdown
import torch
import timm
from torchvision import transforms
from PIL import Image
import streamlit as st

# Load File ID securely from environment variable
FILE_ID = os.getenv("GDRIVE_FILE_ID") or "1GqwTDcFrFmy_FeGm5aZiSEvh3y9B1pMx"
MODEL_PATH = "deepfake_vit_model.pth"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ✅ Download model only if not already present
if not os.path.exists(MODEL_PATH):
    st.info("🔽 Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Failed to download model. Please check the File ID and make sure the file is public.")
        st.stop()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

st.title("🔍 Deepfake Detection using ViT")
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

    label = "Real" if pred == 1 else "Fake"
    color = "green" if label == "Real" else "red"
    st.markdown(f"### Prediction: <span style='color:{color}'>**{label}**</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** `{confidence:.2%}`")

    st.markdown("#### Class Probabilities:")
    st.write({ "Fake": f"{probs[0][0].item():.2%}", "Real": f"{probs[0][1].item():.2%}" })
