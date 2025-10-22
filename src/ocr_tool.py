# src/ocr_tool.py
from typing import List, Tuple
from PIL import Image
import numpy as np

def ocr_easyocr(image: Image.Image, languages: List[str] = ["en"]) -> str:
    import easyocr
    reader = easyocr.Reader(languages, gpu=True)
    np_img = np.array(image.convert("RGB"))
    results = reader.readtext(np_img, detail=0, paragraph=True)
    return "\n".join([r.strip() for r in results if isinstance(r, str) and r.strip()])

def ocr_pytesseract(image: Image.Image) -> str:
    import pytesseract
    return pytesseract.image_to_string(image)