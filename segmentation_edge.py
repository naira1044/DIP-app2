import cv2

def apply_seg_edge(img, task):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_display = gray.copy()
    result = None

    if task == "Basic Global Thresholding":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif task == "Automatic Thresholding (Otsu)":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif task == "Adaptive Thresholding":
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif task == "Sobel Detector":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        result = cv2.convertScaleAbs(magnitude)

    return input_display, result