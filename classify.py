import sys
import torch
from torchvision import models
from PIL import Image

# Load the pretrained model and its matching preprocessing
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

# Load and prepare the image
img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
batch = preprocess(img).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits[0], dim=0)

# Show top 5
top5 = torch.topk(probs, 5)
print("\nTop 5 predictions:")
for p, idx in zip(top5.values, top5.indices):
    print(f"{labels[idx]:30s} {p.item()*100:.2f}%")