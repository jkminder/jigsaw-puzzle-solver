import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import math
from src.filtering import filter
from src.util import *

chien_fn = "chien_rouge_20_1.jpg"
gogh_fn = "van-gogh.png"
craies_fn = "craies_32.png"

example_fn = chien_fn
img = cv2.imread("./../samples/" + example_fn, cv2.IMREAD_COLOR)
orig_width = len(img[0])
orig_height = len(img)

target = 500
scale = target/orig_width if orig_height > orig_width else target/orig_height

resized = cv2.resize(img, None, fx=scale, fy=scale)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = blur(resized)

width = len(resized[0])
height = len(resized)

mask = filter(blurred)

binary_mask = (mask > 0).astype(np.uint8)
# Threshold it so it becomes binary
ret, thresh = cv2.threshold(binary_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type
connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

#remove false detectifons
unique, counts = np.unique(labels, return_counts=True) #is sorted
#remove 0
unique = unique[1:]
counts = counts[1:]

q75, q25 = np.percentile(counts, [75 ,25])
iqr = q75 - q25
low = q25 - 1.5*iqr
hi = q75 + 1.5*iqr

clean_labels = []
clean_counts = []
for i in range(len(counts)):
    if low < counts[i] < hi:
        clean_labels.append(unique[i])
        clean_counts.append(counts[i])

avg = np.average(clean_counts)

margin = 10

boxes = []
for i in range(1,num_labels):
    if i not in clean_labels:
        continue
    x,y = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP]
    w,h = stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
    boxes.append(((x-margin,y-margin), (x+w+margin, y+h+margin)))
    cv2.rectangle(resized, (x-margin,y-margin), (x+w+margin, y+h+margin), 255, 2)


tile_id = 1
## analyse first piece
box = np.multiply(boxes[tile_id],1/scale) #scale box back to original size

crop = img[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
crop_blurred = blur(crop, 21) #TODO param needs to be learned?
mask = filter(crop_blurred)

#filter out image (for edge detection, otherwise the edges in the image itself will be detected)
filtered = np.where(mask[..., None] == 0, crop_blurred ,[255,255,255])
filtered = np.where(mask[..., None] != 0, filtered ,[0,0,0])
edges = cv2.Canny(np.uint8(filtered),10,200, apertureSize=3)

gray = np.float32(mask)
dst = cv2.cornerHarris(gray,20,31,0.04)
dst = cv2.dilate(dst,None)

corners = np.where(dst[..., None]<0.01*dst.max(), filtered,[255,0,0])

plt.subplot(121),plt.imshow(crop,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(corners,cmap = 'gray')
plt.title('Tile mask'), plt.xticks([]), plt.yticks([])

plt.show()
