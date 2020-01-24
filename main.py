import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from src.filtering import filter
from src.util import *
from src.tile_detector import detect_tiles, get_corners
chien_fn = "chien_rouge_20_1.jpg"
gogh_fn = "van-gogh.png"
craies_fn = "craies_32.png"

example_fn = chien_fn
img = cv2.imread("./samples/" + example_fn, cv2.IMREAD_COLOR)

boxes = detect_tiles(img)
tile_id = 1
box = boxes[tile_id]
## analyse first piece
crop = img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]

crop_blurred = blur(crop, 21) #TODO param needs to be learned?
mask = filter(crop_blurred)
mask = mask
#filter out image (for edge detection, otherwise the edges in the image itself will be detected)

v = np.median(mask)

# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
print(lower, upper)

edges = mask - cv2.erode(np.uint8(mask), np.ones((5,5)))

filtered = np.where(edges[..., None] == 0, crop_blurred ,[255,255,255])
filtered = np.where(edges[..., None] != 0, filtered ,[0,0,0])
#edges = cv2.Canny(np.uint8(mask), 0, 255, apertureSize=7)
#edges = cv2.Canny(np.uint8(mask_blurred),10,200, apertureSize=7)

corners = get_corners(mask, 20)

line_mask = np.zeros(edges.shape)
for c1 in corners:
    for c2 in corners:
        pass
        #line_mask = cv2.line(line_mask, c1,c2, 255)

# Blue color in BGR
red = (255, 0, 0)
blue = (0,0,255)

for c in corners:
    crop = cv2.circle(crop, (c[0],c[1]), 4, red, thickness=3)

yxc = ndimage.center_of_mass(mask)
yxc = tuple(np.round(yxc).astype(np.int))
crop = cv2.circle(crop, (yxc[1],yxc[0]), 4, blue, thickness=3)


plt.subplot(121),plt.imshow(crop,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Tile mask'), plt.xticks([]), plt.yticks([])

plt.show()