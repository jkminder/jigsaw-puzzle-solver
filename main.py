import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from src.filtering import filter
from src.util import *
from src.tile_detector import detect_tiles, get_tile_corners, get_angle
from src.image_transformer import four_point_transform
from src.tile import Tile

chien_fn = "chien_rouge_20_1.jpg"
gogh_fn = "van-gogh.png"
craies_fn = "craies_32.png"

example_fn = chien_fn
img = cv2.imread("./samples/" + example_fn, cv2.IMREAD_COLOR)


boxes = detect_tiles(img)


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = np.array(alpha_mask)
    alpha_inv = 1 - np.array(alpha)

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return img

def connect_tiles(tile1, tile2):
    dim = (tile1.img.shape[0]+tile2.img.shape[0], max(tile1.img.shape[1],tile2.img.shape[1]), 3) #TODO someting's wrong here -> find out!
    img = np.zeros((1000,1000,3))
    img = overlay_image_alpha(img, tile1.img, (0,0), tile1.mask)
    x_offset = tile1.corners[1][0]-tile2.corners[0, 0]
    y_offset = 17 # TODO: currently manual tryout -> calculate
    img = overlay_image_alpha(img, tile2.img, (x_offset, y_offset), tile2.mask)
    img = cv2.circle(img, tuple(tile1.corners[1]), 8, green, thickness=6)
    img = cv2.circle(img, (tile2.corners[0][0]+x_offset, tile2.corners[0][1]+y_offset), 17, green, thickness=6)
    return img


for tile_id in range(len(boxes)):
    box = boxes[tile_id]
    crop = img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
    tile = Tile(crop)
    box = boxes[tile_id+2]
    crop = img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
    tile1 = Tile(crop)
    #tile1.show()
    #input()

    test = overlay_image_alpha(np.zeros((1000,1000,3)), np.rot90(tile.img), (0,0), np.rot90(tile.mask))
    test = overlay_image_alpha(test, tile.img, (tile.corners[1][0]-tile.corners[0, 0], 0), tile.mask)
    tile.rotate()
    tile1.rotate(2)
    test = connect_tiles(tile, tile1).astype(int)
    plt.subplot(121),plt.imshow(test ,cmap = 'gray')
    plt.title('original tile with corners'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(tile.img,cmap = 'gray')
    plt.title('perspective correction'), plt.xticks([]), plt.yticks([])
    plt.show()
    input("Press enter for next tile")
