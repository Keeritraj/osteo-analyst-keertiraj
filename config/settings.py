import os
from pathlib import Path

# Base directory is the root of the project (parent of 'config' is root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to critical resources
MODEL_PATH = BASE_DIR / "models" / "densenet121_kl.pth"
ASSETS_DIR = BASE_DIR / "assets"
TEMP_DIR = BASE_DIR / "temp"

# Create temp directory automatically if it doesn't exist
TEMP_DIR.mkdir(exist_ok=True)

# Image Preprocessing Constants (ImageNet Standards)
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Domain Logic: Mapping integers to Medical Labels
KL_GRADES = {
    0: "Healthy",
    1: "Doubtful",
    2: "Minimal",
    3: "Moderate",
    4: "Severe"
}

# GenAI System Prompt
# This instructs the Llama model on how to behave
SYSTEM_PROMPT = (
    "You are an expert Radiologist Assistant. Your task is to generate a preliminary finding "
    "based on the provided Kellgren-Lawrence (KL) Grade. Use professional medical terminology. "
    "Mention 'joint space narrowing', 'osteophyte formation', and 'sclerosis' where appropriate "
    "based on the grade. Keep the response concise and structured as 'Findings:' followed by a paragraph, "
    "then 'Impression:' on a new line."
)