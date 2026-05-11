from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import torch
import cv2

model_name = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name)

model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

def reconhecer_texto(img_crop):

    pil = Image.fromarray(img_crop).convert("RGB")

    pixel_values = processor(
        pil,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = model.generate(pixel_values)

    texto = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return texto