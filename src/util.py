import cv2

# Blue color in BGR
red = (255, 0, 0)
blue = (0,0,255)
green = (0, 255, 0)

def blur(img, filter_size=5):
    return cv2.GaussianBlur(img, (filter_size, filter_size), 0)