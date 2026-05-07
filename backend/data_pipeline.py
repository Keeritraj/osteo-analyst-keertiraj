import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from config.settings import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

def apply_clahe(image: Image.Image) -> Image.Image:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This enhances local contrast (bone edges vs soft tissue) which is critical for 
    detecting Osteoarthritis features like joint space narrowing.
    """
    # Convert PIL to Numpy
    img_np = np.array(image)
    
    # Ensure image is Grayscale (2D array) for CLAHE
    if img_np.ndim == 3:
        # Check if it's RGBA (4 channels) or RGB (3 channels)
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
    # Apply CLAHE
    # clipLimit=2.0 is standard for medical X-rays (avoids amplifying noise)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    
    # Convert back to PIL Image (Mode 'L' = Grayscale)
    return Image.fromarray(enhanced, mode='L')

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Full pipeline: CLAHE -> Resize -> ToTensor -> Normalize.
    Returns a tensor ready for model inference (Batch, C, H, W).
    """
    try:
        # 1. Enhance Contrast
        image = apply_clahe(image).convert("RGB")
        
        # 2. Define Transformations
        # We must convert to RGB because DenseNet/ResNet expects 3 channels
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(), # Converts 0-255 to 0.0-1.0
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        # 3. Apply Transform
        # .convert("RGB") replicates the grayscale channel 3 times
        tensor = transform(image.convert("RGB"))
        
        # 4. Add Batch Dimension (C, H, W) -> (1, C, H, W)
        return tensor.unsqueeze(0)
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")