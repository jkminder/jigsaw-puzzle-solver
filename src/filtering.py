import numpy as np
import cv2
from matplotlib import pyplot as plt


def filter(img):
    # currently only greenscreen filter
    # TODO automatically detect optimal color for background filtering¨
    print("WARING: Green Screen Image is expected. Code cannot handle other types yet.")
    RED, GREEN, BLUE = (2, 1, 0)

    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]

    mask = (greens < 35) | (reds > greens) | (blues > greens)
    return mask
