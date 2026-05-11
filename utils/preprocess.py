import cv2

def preprocess_image(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    thresh = cv2.threshold(
        blur,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    return img, thresh