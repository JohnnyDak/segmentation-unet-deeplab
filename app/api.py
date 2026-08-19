"""
Backend FastAPI : expose les modèles U-Net et DeepLabV3 (chargés depuis
le Model Registry MLflow, statut Production) via une API REST simple.

Usage :
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
import base64
import io

import cv2
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.train import MLFLOW_TRACKING_URI

IMAGE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Segmentation de lésions cutanées — API")

MODELS: dict = {}


@app.on_event("startup")
def load_models() -> None:
    print("Chargement des modèles depuis le Model Registry MLflow...")
    MODELS["unet"] = mlflow.pytorch.load_model(
        "models:/unet-lesion-segmentation/Production"
    ).to(DEVICE).eval()
    MODELS["deeplabv3"] = mlflow.pytorch.load_model(
        "models:/deeplabv3-lesion-segmentation/Production"
    ).to(DEVICE).eval()
    print("Modèles chargés :", list(MODELS.keys()))


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Même redimensionnement/normalisation qu'à l'entraînement."""
    image = np.array(pil_image.convert("RGB"))
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def mask_to_base64_png(mask: np.ndarray) -> str:
    """Encode un masque binaire (0/1) en PNG base64, transportable en JSON."""
    mask_img = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", mask_img)
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Reçoit une image, renvoie les masques prédits par les deux
    architectures (encodés en PNG base64)."""
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    tensor = preprocess(pil_image)

    masks = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            logits = model(tensor)
            mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
            masks[name] = mask_to_base64_png(mask)

    return JSONResponse(content={"masks": masks, "image_size": IMAGE_SIZE})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)