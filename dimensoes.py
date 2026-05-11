import cv2
from pathlib import Path

path_atual = Path.cwd()
path_imagens = path_atual / 'images' / 'imagens_padronizadas'
img = cv2.imread(str(path_imagens / "pagina_1.png"))

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE:
        print(f"X={x} Y={y}")

cv2.namedWindow("Imagem")

cv2.setMouseCallback(
    "Imagem",
    mouse_callback
)

while True:

    cv2.imshow("Imagem", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()