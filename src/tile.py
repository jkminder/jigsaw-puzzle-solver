import numpy as np
from matplotlib import pyplot as plt
import cv2

from src.util import blur, red, blue, green
from src.filtering import filter
from src.tile_detector import get_tile_corners
from src.image_transformer import four_point_transform


class Tile:

    def __init__(self, tile_img):
        self.orig_img = tile_img
        self.orig_mask = filter(blur(self.orig_img, 21))
        self.orig_corners = np.array(get_tile_corners(self.orig_mask))
        self.img, self.corners = four_point_transform(self.orig_img, self.orig_corners)
        self.mask, _ = four_point_transform(np.float32(self.orig_mask), self.orig_corners)

    def show(self):
        img = self.img
        for c in self.corners:
            img = cv2.circle(img, tuple(c), 8, green, thickness=3)

        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('tile'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.mask, cmap='gray')
        plt.title('tile mask'), plt.xticks([]), plt.yticks([])
        plt.show()


    def rotate(self, num=1):
        for i in range(num):
            self.img = np.rot90(self.img)
            self.mask = np.rot90(self.mask)

