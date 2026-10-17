import streamlit as st
import torch
from torchvision import models
from PIL import Image

@st.cache_resource
def load_model():
    w = models.ResNet50_Weights.DEFAULT
    m = models.resnet50(weights=w).eval()
    return m, w.transforms(), w.meta["categories"]

model, preprocess, labels = load_model()

st.title("Image Classifier (ResNet50)")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img).unsqueeze(0))[0], dim=0)
    top5 = torch.topk(probs, 5)
    for p, i in zip(top5.values, top5.indices):
        st.write(f"**{labels[i]}**: {p.item()*100:.2f}%")
        st.progress(float(p))