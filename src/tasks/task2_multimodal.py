"""
task2_multimodal.py - Multi-modal image chat using Gemini vision.
"""

import google.generativeai as genai
from PIL import Image

from config import GOOGLE_API_KEY, VISION_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


def describe_image(image: Image.Image, question: str = "") -> str:
    model = genai.GenerativeModel(VISION_MODEL)
    prompt = question.strip() or "Describe this image in detail."
    response = model.generate_content([prompt, image])
    return response.text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python task2_multimodal.py <image_path> [question]")
    else:
        img = Image.open(sys.argv[1])
        q = sys.argv[2] if len(sys.argv) > 2 else ""
        print(describe_image(img, q))
