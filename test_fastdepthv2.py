import torch
from models.models import FastDepthV2

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
WEIGHTS_PATH = "Weights/FastDepthV2_L1GN_Best.pth"
IMAGE_PATH = "sample.jpg"  # Replace with your image path
USE_CUDA = torch.cuda.is_available()

# --- Load model ---
device = torch.device("cuda" if USE_CUDA else "cpu")
model = FastDepthV2()
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# --- Load image ---
input_image = Image.open(IMAGE_PATH).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(input_image).unsqueeze(0).to(device)

# --- Inference ---
with torch.no_grad():
    output = model(input_tensor)
    depth_map = output.squeeze().cpu().numpy()

# --- Show result ---
plt.imshow(depth_map, cmap='inferno')
plt.colorbar()
plt.title("Predicted Depth Map")
plt.axis('off')
plt.show()
