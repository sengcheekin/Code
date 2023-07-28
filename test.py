import skyDetector as detector
import cv2 as cv
from matplotlib import pyplot as plt 
import os
import numpy as np

def turn_subsequent_pixels_to_zero(image):
    h, w = image.shape
    for col in range(w):
        for row in range(h):
            if image[row, col] == 0:
                image[row:, col] = 0
                break
    return image

def get_skyline(img):
    dilated = cv.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
    skyline = dilated - img

    return skyline

def evaluate(output, ground_truth):
    diff = cv.absdiff(output, ground_truth)
    diff = diff.astype(np.uint8)
    percentage = (np.count_nonzero(diff == 0) / diff.size) * 100

    return percentage




if __name__ == "__main__":

    data_folder = "dataset/test"

    for img_folder in os.listdir(data_folder):
        
        ground_truth = cv.imread(f'dataset/ground_truth/{img_folder}_GT.png', 0)
        # convert to binary for evaluation later
        ground_truth = cv.threshold(ground_truth, 127, 1, cv.THRESH_BINARY)[1]

        for img in os.listdir(os.path.join(data_folder, img_folder)):
            print(img)

            img = cv.imread(os.path.join(data_folder, img_folder, img), 0)

            img_sky = detector.get_sky_region_gradient(img)

            # plt.figure(2)
            # plt.subplot(2,1,1)
            # plt.imshow(img_sky)
            # plt.title('sky 1')

            # can try with rectangular kernel
            img_sky = cv.morphologyEx(img_sky, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            img_sky = cv.erode (img_sky, np.ones((7, 7), np.uint8), iterations=1)
            
            img_sky = turn_subsequent_pixels_to_zero(img_sky)

            plt.subplot(2,1,1)
            plt.imshow(img_sky)
            plt.title('result')

            plt.subplot(2,1,2)
            plt.imshow(ground_truth)
            plt.title('ground truth')
            plt.show()

            # print type
            print(img_sky.dtype, ground_truth.dtype)

            similarity = evaluate(img_sky, ground_truth)
            print(similarity)

            # img_skyline = get_skyline(img_sky)
            # plt.figure(3)
            # plt.imshow(img_skyline)
            # plt.title('skyline')
            # plt.show()
