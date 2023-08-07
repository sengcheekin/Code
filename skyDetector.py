# Code taken and modified from https://github.com/cftang0827/sky-detector/blob/master/sky_detector/detector.py

import cv2
from scipy.signal import medfilt
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt


def cal_skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]

            if first_zero_index > 20:
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask

# Modified to expect a grayscale image instead of a BGR image, and to return the mask instead of the image.
def get_sky_region_gradient(img):

    img = cv2.blur(img, (9, 3))
    cv2.medianBlur(img, 5)
    lap = cv2.Laplacian(img, cv2.CV_8U)
    gradient_mask = (lap < 6).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)
    # after_img = cv2.bitwise_and(img, img, mask=mask)

    return mask