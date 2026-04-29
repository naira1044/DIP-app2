import cv2
import numpy as np

def apply_point_color(img, task):
    input_display = img.copy()
    result = img.copy()

    # Point Operations
    if task == "Addition":
        val = np.array([50.0, 50.0, 50.0])
        result = cv2.add(img, val, dtype=cv2.CV_8U)
    elif task == "Subtraction":
        val = np.array([50.0, 50.0, 50.0])
        result = cv2.subtract(img, val, dtype=cv2.CV_8U)
    elif task == "Division":
        result = cv2.divide(img, 2.0)
    elif task == "Complement":
        result = cv2.bitwise_not(img)
        
    # Color Operations
    elif task == "Change Red Lighting":
        result[:, :, 2] = cv2.add(result[:, :, 2], 50)
    elif task == "Swap R to G":
        temp = result[:, :, 1].copy()
        result[:, :, 1] = result[:, :, 2]
        result[:, :, 2] = temp
    elif task == "Eliminate Red":
        result[:, :, 2] = 0

    return input_display, result