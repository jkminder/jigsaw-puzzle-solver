import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils


def filter(img):
    # currently only greenscreen filter
    # TODO automatically detect optimal color for background filtering¨
    print("WARING: Green Screen Image is expected. Code cannot handle other types yet.")
    RED, GREEN, BLUE = (2, 1, 0)

    empty_img = np.zeros_like(img)

    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]

    mask = (greens < 35) | (reds > greens) | (blues > greens)
    empty_img[(greens < 35) | ((reds <= greens) & (blues <= greens))][BLUE] = 255
    return mask
