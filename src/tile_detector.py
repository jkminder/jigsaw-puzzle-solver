import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from src.filtering import filter
from src.util import *

chien_fn = "chien_rouge_20_1.jpg"
gogh_fn = "van-gogh.png"
craies_fn = "craies_32.png"

example_fn = chien_fn
img = cv2.imread("./../samples/" + example_fn, cv2.IMREAD_COLOR)

def detect_tiles(img, margin=10):
    orig_width = len(img[0])
    orig_height = len(img)

    target = 500
    scale = target/orig_width if orig_height > orig_width else target/orig_height

    resized = cv2.resize(img, None, fx=scale, fy=scale)
    blurred = blur(resized)
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
    #remove 0 (background)
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

    boxes = []
    for i in range(1,num_labels):
        if i not in clean_labels:
            continue
        x, y = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP]
        w, h = stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
        box = ((x-(h-w)/2-margin,y-margin), (x+h+margin-(h-w)/2, y+h+margin)) if h > w else \
            ((x - margin, y - margin - (w-h) / 2), (x + w + margin, y + w + margin - (w-h) / 2))
        boxes.append(np.multiply(box, 1/scale))

    return boxes

def get_corners(mask, harris_blocksize):
    gray = np.float32(mask)
    dst = cv2.cornerHarris(gray, harris_blocksize, 31, 0.04)
    dst = cv2.dilate(dst, None)
    dst = dst * gray
    data = dst.copy()
    data[data < 0.4 * data.max()] = 0
    datamax = filters.maximum_filter(data, 5)
    maxima = (dst == datamax)
    datamin = filters.minimum_filter(data, 5)
    minima = (dst == datamin)
    diff = ((datamax - datamin) > .01)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    # slices = ndimage.find_objects(labeled)
    # yx = np.array(ndimage.center_of_mass(dst, labeled, range(1, num_objects+1)))
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
    temp = yx[:, 0].copy()
    yx[:, 0] = yx[:, 1]
    yx[:, 1] = temp
    return np.round(yx).astype(np.int)

def get_vector(p1,p2):
    if type(p1).__module__ != np.__name__:
        p1 = np.array(p1)
    if type(p2).__module__ != np.__name__:
        p2 = np.array(p2)
    return p2-p1


def get_angle(p1,p2,p3):
    """calculate angle between p2_p3 and p2_p3"""
    p2p1 = get_vector(p2,p1)
    p2p3 = get_vector(p2,p3)
    cosine_angle = np.dot(p2p1, p2p3) / (np.linalg.norm(p2p1) * np.linalg.norm(p2p3))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_90deg_corners(pt, corners, rule1, rule2, margin = 5):
    """calculate all corners that are 90 degrees from pt, where the corners c1, c2 must comply with rule1, rule2 """
    res = []
    used = []
    for c1 in corners:
        if not rule1(c1):
            continue
        for c2 in corners:
            if np.array_equal(c1,c2) or np.array_equal(c2,pt) or np.array_equal(c1,pt) or \
                    not rule2(c2) or \
                    tuple(c2) in used:
                continue
            if 90 - margin < get_angle(c1,pt,c2) < 90 + margin:
                res.append((c1,c2))
                used.append(tuple(c1))
    return res


def get_tile_corners(corners, tile_center, angle_margin=10):
    tile_corners = []
    for c1 in corners:
        if c1[0] <= tile_center[0] and c1[1] <= tile_center[1]:
            # identify candidates for top left corner

            candidates1 = get_90deg_corners(c1, corners,
                                            lambda c: c[0] <= tile_center[0] and c[1] >= tile_center[1],
                                            lambda c: c[0] >= tile_center[0] and c[1] <= tile_center[1], angle_margin)

            for c2, c4 in candidates1:
                for c3 in corners:
                    if c3[0] >= tile_center[0] and c3[1] >= tile_center[1]:
                        # identify candidates for bottom right corner
                        candidates2 = get_90deg_corners(c3, corners,
                                                        lambda c: True,
                                                        # c[0] <= tile_center[0] and c[1] >= tile_center[1],
                                                        lambda c: True,
                                                        angle_margin)  # c[0] >= tile_center[0] and c[1] <= tile_center[1])
                        for t2, t4 in candidates2:
                            if (((np.array_equal(c2, t2) and np.array_equal(c4, t4)) or
                                 (np.array_equal(c2, t4) and np.array_equal(c4, t2)))) and 90 - angle_margin < get_angle(c2,c3,c4) < 90 + angle_margin and 90 - angle_margin < get_angle(c3, c4, c1) < 90 + angle_margin:
                                if len(tile_corners) > 0:
                                    # check if better match
                                    opt = np.array([90, 90, 90, 90])
                                    ang_new = np.array(
                                        [get_angle(c4, c1, c2), get_angle(c1, c2, c3), get_angle(c2, c3, c4),
                                         get_angle(c3, c4, c1)])
                                    ang_curr = np.array([get_angle(tile_corners[3], tile_corners[0], tile_corners[1]),
                                                         get_angle(tile_corners[0], tile_corners[1], tile_corners[2]),
                                                         get_angle(tile_corners[1], tile_corners[2], tile_corners[3]),
                                                         get_angle(tile_corners[2], tile_corners[3], tile_corners[0])])
                                    diff_new = np.sum(np.square(opt - ang_new))
                                    diff_curr = np.sum(np.square(opt - ang_curr))
                                    if diff_new > diff_curr:
                                        continue
                                tile_corners = [c1, c2, c3, c4]
    return tile_corners