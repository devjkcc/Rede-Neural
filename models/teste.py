import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CARREGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "modelo_checkbox.keras"
)

# =========================
# IMAGEM TESTE
# =========================

img = cv2.imread("debug_dias_recebimento_Ter.png")

# =========================
# PREPROCESSAMENTO
# =========================

# BGR -> GRAYSCALE
gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# Resize MobileNet
gray = cv2.resize(
    gray,
    (96, 96)
)

# GRAYSCALE -> RGB
rgb = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2RGB
)

# float32
rgb = rgb.astype(np.float32)

# preprocessamento MobileNet
rgb = preprocess_input(rgb)

# batch dimension
rgb = np.expand_dims(rgb, axis=0)

# =========================
# PREVISÃO
# =========================

pred = model.predict(
    rgb,
    verbose=0
)

prob = pred[0][0]

print("Probabilidade:", prob)

# =========================
# RESULTADO
# =========================

if prob < 0.05:
    print("✓ Marcada")
else:
    print("✗ Vazia")