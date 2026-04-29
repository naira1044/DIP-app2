import cv2
import numpy as np

def apply_filters_noise(img, task):
    input_display = img.copy()
    result = None

    # Filters
    if task == "Linear: Average Filter":
        result = cv2.blur(img, (7, 7))
    elif task == "Linear: Laplacian Filter":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_display = gray.copy()
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result = cv2.convertScaleAbs(laplacian)
    elif task == "Non-linear: Maximum":
        kernel = np.ones((5,5), np.uint8)
        result = cv2.dilate(img, kernel, iterations=1)
    elif task == "Non-linear: Minimum":
        kernel = np.ones((5,5), np.uint8)
        result = cv2.erode(img, kernel, iterations=1)
    elif task == "Non-linear: Median":
        result = cv2.medianBlur(img, 7)
    elif task == "Non-linear: Mode (Most Frequent)":
        from scipy.ndimage import generic_filter
        from scipy import stats
        def mode_func(x):
            return stats.mode(x, keepdims=False)[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_display = gray.copy()
        small = cv2.resize(gray, (200, 200)) # Downscale for performance
        res_small = generic_filter(small, mode_func, size=3)
        result = cv2.resize(res_small, (gray.shape[1], gray.shape[0]))

    # Noise & Restoration
    elif task.startswith("Salt & Pepper"):
        noisy = img.copy()
        prob = 0.05
        thres = 1 - prob
        rand_matrix = np.random.rand(img.shape[0], img.shape[1])
        noisy[rand_matrix < prob] = 0
        noisy[rand_matrix > thres] = 255
        input_display = noisy.copy()
        
        if "Average" in task:
            result = cv2.blur(noisy, (5, 5))
        elif "Median" in task:
            result = cv2.medianBlur(noisy, 5)
        elif "Outlier" in task:
            median = cv2.medianBlur(noisy, 5)
            diff = cv2.absdiff(noisy, median)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = np.where(mask_bgr == 255, median, noisy)
            
    elif task.startswith("Gaussian"):
        if "Averaging" in task:
            noisy_images = []
            for _ in range(10):
                noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                noisy_images.append(noisy_img)
            input_display = noisy_images[0] 
            result = np.mean(noisy_images, axis=0).astype(np.uint8)
        else:
            noise = np.random.normal(0, 25, img.shape).astype(np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            input_display = noisy.copy()
            result = cv2.blur(noisy, (5, 5))

    return input_display, result