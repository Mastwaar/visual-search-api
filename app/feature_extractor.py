import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_image_features(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(x)
    return feats.squeeze().cpu().numpy()
