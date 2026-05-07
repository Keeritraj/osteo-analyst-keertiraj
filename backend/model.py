import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import models
from config.settings import MODEL_PATH

class OsteoClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Load standard DenseNet121 architecture
        self.backbone = models.densenet121(weights=None)
        
        # Get the number of input features going into the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace the default classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # CRITICAL FIX FOR GRAD-CAM:
        # We manually run the DenseNet steps to ensure ReLU is NOT done in-place.
        # Standard self.backbone(x) uses inplace=True which breaks Grad-CAM hooks.
        
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False)  # <--- The Magic Fix
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(out)
        return out

def load_model():
    """
    Robust model loader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OsteoClassifier()
    
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Load failed ({e}). Using random weights.")
    else:
        print("⚠️ Model file missing or empty. Using random weights.")

    model.to(device)
    model.eval()
    return model

def predict(model, input_tensor):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
    return pred_class, confidence

def generate_grad_cam(model, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap.
    """
    try:
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        gradients = None
        activations = None

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach() # detach is safer than clone here given the forward fix

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        # Hook the last layer of features
        target_layer = model.backbone.features.norm5
        
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward
        logits = model(input_tensor)
        
        # Backward
        model.zero_grad()
        logits[0, target_class].backward()
        
        h1.remove()
        h2.remove()
        
        if gradients is None or activations is None:
            return np.zeros((224, 224))

        # Weights: [1, 1024, 1, 1]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # CAM: [1, 7, 7]
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU & Normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = cam.cpu().numpy()
        return cv2.resize(cam, (224, 224))

    except Exception as e:
        print(f"❌ Grad-CAM Error: {e}")
        return np.zeros((224, 224))