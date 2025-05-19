import torch
import cv2
import numpy as np
from models.models import FastDepthV2
from load_pretrained import load_pretrained_fastdepth

# Paths
weights_path = "Weights/FastDepthV2_L1GN_Best.pth"
image_path = "test.jpg"  # Change this to your image

# Load image and preprocess
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FastDepthV2()
model, _ = load_pretrained_fastdepth(model, weights_path)
model = model.to(device)
model.eval()

# Inference
with torch.no_grad():
    output = model(tensor.to(device)).squeeze().cpu().numpy()

# Normalize and save
norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("depth_map.png", norm.astype(np.uint8))
print("✅ Depth map saved as depth_map.png")
