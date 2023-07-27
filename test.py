import skyDetector as detector
import cv2
from matplotlib import pyplot as plt 
import os

img_folder = "dataset/day"
print(os.listdir(img_folder))

for img_name in os.listdir(img_folder):

    img = cv2.imread(os.path.join(img_folder, img_name))[:,:,::-1] 
    # plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.imshow(img)

    img_sky = detector.get_sky_region_gradient(img)
    # convert image to binary
    # img_sky = cv2.cvtColor(img_sky, cv2.COLOR_RGB2GRAY)
    ret, img_sky = cv2.threshold(img_sky, 0, 255, cv2.THRESH_BINARY)

    plt.figure(2)
    plt.subplot(2,1,2)
    plt.imshow(img_sky)
    plt.show()

