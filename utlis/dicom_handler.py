import pydicom
import numpy as np
from PIL import Image
import io

def dicom_to_pil(dicom_source):
    """
    Convert DICOM file (path or file-like object) to PIL Image (grayscale).
    Handles bit-depth normalization and basic photometric interpretation.
    """
    try:
        # Check if input is a file path or a stream (BytesIO from Streamlit)
        if isinstance(dicom_source, (str, Path)):
            dicom = pydicom.dcmread(dicom_source)
        else:
            # It's a file-like object from Streamlit
            dicom = pydicom.dcmread(dicom_source)

        pixel_array = dicom.pixel_array

        # 1. Handle Photometric Interpretation (Monochrome1 vs Monochrome2)
        # MONOCHROME1: 0=White, 1=Black (Inverted)
        # MONOCHROME2: 0=Black, 1=White (Standard)
        if hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = np.amax(pixel_array) - pixel_array

        # 2. Normalize to 8-bit (0-255)
        # Medical images can be 12-bit or 16-bit
        if pixel_array.dtype != np.uint8:
            pixel_array = pixel_array.astype(np.float32)
            # Min-Max Normalization
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
            # Scale to 255
            pixel_array = (pixel_array * 255).astype(np.uint8)

        # 3. Convert to PIL Image
        image = Image.fromarray(pixel_array, mode='L')
        return image

    except Exception as e:
        # Re-raise with a clear error message for the UI
        raise ValueError(f"Failed to process DICOM file: {str(e)}")