import cv2
import numpy as np

def apply_morphology(img, task):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    input_display = binary.copy()
    kernel = np.ones((5,5), np.uint8)
    result = None

    if task == "Image Dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif task == "Image Erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    elif task == "Image Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif task == "Boundary: Internal":
        erosion = cv2.erode(binary, kernel, iterations=1)
        result = cv2.subtract(binary, erosion)
    elif task == "Boundary: External":
        dilation = cv2.dilate(binary, kernel, iterations=1)
        result = cv2.subtract(dilation, binary)
    elif task == "Boundary: Morphological Gradient":
        result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    return input_display, result