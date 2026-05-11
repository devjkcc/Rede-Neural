import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "modelo_checkbox.h5"
)

def checkbox_marcado_ml(img_crop):

    img = cv2.cvtColor(
        img_crop,
        cv2.COLOR_BGR2GRAY
    )

    img = cv2.resize(img, (32,32))

    img = img / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    return pred[0][0] < 0.2