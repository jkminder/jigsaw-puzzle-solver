import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from src.filtering import filter
from src.util import *
from src.tile_detector import detect_tiles, get_tile_corners
chien_fn = "chien_rouge_20_1.jpg"
gogh_fn = "van-gogh.png"
craies_fn = "craies_32.png"

example_fn = chien_fn
img = cv2.imread("./samples/" + example_fn, cv2.IMREAD_COLOR)

# Blue color in BGR
red = (255, 0, 0)
blue = (0,0,255)
green = (0, 255, 0)

boxes = detect_tiles(img)
"""
10,12 - edges are not all detected
13 - problem with a different square
"""
for tile_id in range(len(boxes)):
    print(tile_id)
    box = boxes[tile_id]
    ## analyse first piece
    crop = img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]

    crop_blurred = blur(crop, 21) #TODO param needs to be learned?
    mask = filter(crop_blurred)
    #filter out image (for edge detection, otherwise the edges in the image itself will be detected)

    v = np.median(mask)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))

    edges = mask - cv2.erode(np.uint8(mask), np.ones((5,5)))

    filtered = np.where(edges[..., None] == 0, crop_blurred ,[255,255,255])
    filtered = np.where(edges[..., None] != 0, filtered ,[0,0,0])

    #detect the tile edges
    tile_corners, crop = get_tile_corners(mask,crop)

    if len(tile_corners) >= 3:
        crop = cv2.line(crop, tuple(tile_corners[0]), tuple(tile_corners[3]), red, 4)
        crop = cv2.line(crop, tuple(tile_corners[1]), tuple(tile_corners[0]), red, 4)
        crop = cv2.line(crop, tuple(tile_corners[2]), tuple(tile_corners[1]), red, 4)
        crop = cv2.line(crop, tuple(tile_corners[3]), tuple(tile_corners[2]), red, 4)
    for c in tile_corners:
        crop = cv2.circle(crop, tuple(c), 8, green, thickness=3)



    plt.subplot(121),plt.imshow(crop ,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(mask,cmap = 'gray')
    plt.title('Tile mask'), plt.xticks([]), plt.yticks([])

    plt.show()
    input("Press enter for next tile")