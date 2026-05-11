import easyocr
import re

# carrega uma vez só
reader = easyocr.Reader(
    ['pt'],
    gpu=False
)

def reconhecer_numeros(img):

    resultados = reader.readtext(
        img,
        detail=0,
        paragraph=False,
        allowlist='0123456789()-./ '
    )

    texto = " ".join(resultados)

    # limpa caracteres indesejados
    texto = re.sub(
        r'[^0-9()/\-. ]',
        '',
        texto
    )

    texto = texto.strip()

    return texto