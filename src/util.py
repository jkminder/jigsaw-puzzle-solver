import cv2

def blur(img, filter_size=5):
    return cv2.GaussianBlur(img, (filter_size, filter_size), 0)