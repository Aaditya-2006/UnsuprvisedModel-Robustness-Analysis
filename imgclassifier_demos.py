import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

try:
    from imagecorruptions import corrupt
    HAS_CORRUPTION_LIB = True
except ImportError:
    HAS_CORRUPTION_LIB = False
    print("Note: 'imagecorruptions' library not found. Using simple noise fallback.")

MODEL_DIR = 'saved_models' 
DEVICE = torch.device('cpu')
CLASSES = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class selfsupervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.shape[0], -1)
        z = self.projection_head(h)
        return h, z

class RobustClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            h, _ = self.encoder(x)
        return self.fc(h)

def get_supervised_model():
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

print(f"Loading models from")

backbone_path = os.path.join(MODEL_DIR, 'simclr_backbone.pth')
if not os.path.exists(backbone_path):
    print(f"Error: Backbone not found")
    sys.exit(1)

simclr_framework = selfsupervised()
simclr_framework.load_state_dict(torch.load(backbone_path, map_location=DEVICE))

simclr_model = RobustClassifier(simclr_framework)
head_path = os.path.join(MODEL_DIR, 'simclr_classifier.pth')

if os.path.exists(head_path):
    simclr_model.fc.load_state_dict(torch.load(head_path, map_location=DEVICE))
    simclr_model.eval()
    print("SimCLR Model loaded.")
else:
    print(f"Classifier not found")

sup_path = os.path.join(MODEL_DIR, 'supervised_baseline.pth')
if os.path.exists(sup_path):
    sup_model = get_supervised_model()
    sup_model.load_state_dict(torch.load(sup_path, map_location=DEVICE))
    sup_model.eval()
    print("Supervised Model loaded.")
else:
    sup_model = None
    print("Supervised baseline not found.")


inference_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_prediction(model, tensor):
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        return CLASSES[pred.item()], conf.item() * 100

def predict_local(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    try:
        original_img = Image.open(image_path).convert('RGB')
        
        plt.figure(figsize=(4,4))
        plt.imshow(original_img)
        plt.axis('off')
        plt.title("Original Image")
        plt.show()
        
        clean_tensor = inference_transform(original_img).unsqueeze(0).to(DEVICE)
        
        sim_pred, sim_conf = get_prediction(simclr_model, clean_tensor)
        if sup_model:
            sup_pred, sup_conf = get_prediction(sup_model, clean_tensor)
        else:
            sup_pred, sup_conf = "N/A", 0.0

        print(f"CLEAN RESULTS:")
        print(f"   Unsupervised Results:      {sim_pred} ({sim_conf:.1f}%)")
        print(f"   Supervised Results:        {sup_pred} ({sup_conf:.1f}%)")

        img_np = np.array(original_img.resize((32, 32)))
        
        if HAS_CORRUPTION_LIB:
            noisy_np = corrupt(img_np, severity=1, corruption_name='defocus_blur')
        else:
            noisy_np = img_np + np.random.normal(0, 25, img_np.shape).astype(np.uint8)
            noisy_np = np.clip(noisy_np, 0, 255)
            
        noisy_img = Image.fromarray(noisy_np)

        plt.figure(figsize=(4,4))
        plt.imshow(noisy_np)
        plt.axis('off')
        plt.title("Noisy Image (Severity 1)")
        plt.show()

        noisy_tensor = inference_transform(noisy_img).unsqueeze(0).to(DEVICE)
        
        sim_noisy_pred, sim_noisy_conf = get_prediction(simclr_model, noisy_tensor)
        if sup_model:
            sup_noisy_pred, sup_noisy_conf = get_prediction(sup_model, noisy_tensor)
        else:
            sup_noisy_pred, sup_noisy_conf = "N/A", 0.0

        print(f"NOISY RESULTS:")
        print(f"   Unsupervised Results:      {sim_noisy_pred} ({sim_noisy_conf:.1f}%)")
        print(f"   Supervised Results:        {sup_noisy_pred} ({sup_noisy_conf:.1f}%)")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print(f"\nReady. Drop an image path below (or type 'exit'):")
    while True:
        user_input = input(">> ").strip().strip('"')
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input:
            predict_local(user_input)