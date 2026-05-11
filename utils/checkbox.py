import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

path_atual = Path.cwd()
path_anterior_1 = path_atual.parent

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# CARREGAR MODELO
# =========================

model = tf.keras.models.load_model(
    BASE_DIR / "models" / "modelo_checkbox.keras"
)

# =========================
# FUNÇÃO
# =========================

def checkbox_marcado(img_crop):

    # BGR -> GRAYSCALE
    img = cv2.cvtColor(
        img_crop,
        cv2.COLOR_BGR2GRAY
    )

    img = cv2.resize(
    img,
    None,
    fx=4,
    fy=4,
    interpolation=cv2.INTER_CUBIC
)

    # Resize novo da MobileNet
    img = cv2.resize(img, (96, 96))

    # GRAYSCALE -> RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2RGB
    )

    # float32
    img = img.astype(np.float32)

    # preprocessamento MobileNetV2
    img = preprocess_input(img)

    # batch dimension
    img = np.expand_dims(img, axis=0)

    # predição
    pred = model.predict(
        img,
        verbose=0
    )

    prob = pred[0][0]

    print("Probabilidade:", prob)

    # Ajuste dependendo da classe positiva
    return prob < 0.05