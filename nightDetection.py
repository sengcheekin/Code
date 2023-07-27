# Find brightness of image and compare it to a threshold value
# If the brightness is below the threshold, it is night
# If the brightness is above the threshold, it is day

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
# img = cv2.imread('dataset/623/20120901_084151.jpg') # night image
# img = cv2.imread('dataset/623/20120901_174155.jpg') # day image
# img = cv2.imread('dataset/623/20120905_004156.jpg') # dawn image
img = cv2.imread('dataset/9730/20130329_052708.jpg') # dusk image

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find brightness of image
brightness = np.mean(gray)

# Compare brightness to threshold
if brightness < 100:
    print("It is night")
else:
    print("It is day")

# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()