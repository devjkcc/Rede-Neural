import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# =========================
# CAMINHOS
# =========================

path_atual = Path.cwd()
path_anterior_1 = path_atual.parent

caminho_imagens = path_anterior_1 / 'images' / 'Treinamento'

# =========================
# CONFIG
# =========================

IMG_SIZE = (96, 96)
BATCH_SIZE = 16

# =========================
# DATASETS
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    caminho_imagens,
    validation_split=0.2,
    subset="training",
    seed=123,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    caminho_imagens,
    validation_split=0.2,
    subset="validation",
    seed=123,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# CONVERTER GRAYSCALE -> RGB
# =========================

def grayscale_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_ds = train_ds.map(grayscale_to_rgb)
val_ds = val_ds.map(grayscale_to_rgb)

# =========================
# PREPROCESSAMENTO MOBILENET
# =========================

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# Melhor performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# =========================
# BASE MODEL - MOBILENETV2
# =========================

base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)

# IMPORTANTE:
# Mantém MobileNet congelada
# Ideal para dataset pequeno

base_model.trainable = False

# =========================
# CALLBACKS
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# MODELO
# =========================

model = models.Sequential([

    # =========================
    # DATA AUGMENTATION
    # =========================

    layers.RandomRotation(0.03),

    layers.RandomZoom(0.05),

    layers.RandomTranslation(
        height_factor=0.03,
        width_factor=0.03
    ),

    # =========================
    # MOBILENETV2
    # =========================

    base_model,

    # =========================
    # CLASSIFICADOR
    # =========================

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')
])

# =========================
# COMPILAR
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TREINAR
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# =========================
# SALVAR
# =========================

model.save("modelo_checkbox.keras")

print("✓ Modelo treinado!")