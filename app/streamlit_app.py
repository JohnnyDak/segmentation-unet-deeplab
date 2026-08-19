"""
Frontend Streamlit : upload d'image -> appelle l'API FastAPI (/predict)
-> affiche les masques prédits par les deux architectures, superposés
sur l'image, côte à côte pour comparaison directe.

Suppose que le backend FastAPI tourne déjà sur http://localhost:8000
(voir app/api.py).

Usage :
    streamlit run app/streamlit_app.py
"""
import base64
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Segmentation de lésions cutanées", layout="wide")
st.title("Segmentation de lésions cutanées — U-Net vs DeepLabV3")
st.markdown(
    "Upload une image de lésion cutanée (dataset ISIC). Les deux modèles, "
    "entraînés from scratch, prédisent chacun un masque de segmentation."
)

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = ((1 - alpha) * overlay[mask > 0] + alpha * np.array(color)).astype(np.uint8)
    return overlay


def decode_mask(b64_string: str) -> np.ndarray:
    mask_bytes = base64.b64decode(b64_string)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    return (np.array(mask_img) > 127).astype(np.uint8)


if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Prédiction en cours (les deux modèles)..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(API_URL, files=files, timeout=30)
        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de joindre l'API FastAPI sur "
                f"{API_URL}. Vérifie qu'elle tourne bien (uvicorn app.api:app --port 8000)."
            )
            st.stop()

    if response.status_code != 200:
        st.error(f"Erreur API : {response.status_code} — {response.text}")
    else:
        data = response.json()
        image_size = data["image_size"]
        display_image = np.array(pil_image.resize((image_size, image_size)))

        unet_mask = decode_mask(data["masks"]["unet"])
        deeplab_mask = decode_mask(data["masks"]["deeplabv3"])

        unet_overlay = overlay_mask(display_image, unet_mask, color=(255, 0, 0))
        deeplab_overlay = overlay_mask(display_image, deeplab_mask, color=(0, 100, 255))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(display_image, caption="Image originale (redimensionnée)")
        with col2:
            st.image(unet_overlay, caption="U-Net — masque prédit (rouge)")
        with col3:
            st.image(deeplab_overlay, caption="DeepLabV3 — masque prédit (bleu)")