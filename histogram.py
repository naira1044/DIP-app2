import cv2
import numpy as np

def apply_histogram(img, task):
    # تحويل الصورة لرمادي كما هو مطلوب في المهام 8 و 9
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_display = gray.copy()
    result = None

    if task == "Histogram Stretching (Gray)":
        # تطبيق المعادلة الموجودة في Sec 4 - Page 16/17
        I_low = np.min(gray)
        I_high = np.max(gray)
        
        # لمنع القسمة على صفر في حالة الصور الموحدة
        if I_high - I_low == 0:
            result = gray
        else:
            # المعادلة: (Pixel - Min) * (255 / (Max - Min))
            result = ((gray - I_low) / (I_high - I_low))*255.0
            result = result.astype(np.uint8)

    elif task == "Histogram Equalization (Gray)":
        # 1. Calculate histogram
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        
        # 2. Calculate PDF (Normalized Histogram)
        pdf = hist / gray.size
        
        # 3. Calculate CDF (Cumulative Density Function)
        cdf = pdf.cumsum()
        
        # 4. Calculate the equalized mapping (Multiply by L-1, which is 255)
        # We use rounding to get integer pixel values
        mapping = np.round(cdf * 255).astype(np.uint8)
        
        # 5. Apply the mapping to the original image pixels
        result = mapping[gray]

    return input_display, result