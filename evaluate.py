# Code to find how similar the detected sky region is compared to the ground truth

import cv2 as cv
import numpy as np

def evaluate(output, ground_truth):
    # diff = cv.absdiff(output, ground_truth)
    diff = np.abs(output - ground_truth)
    diff = diff.astype(np.uint8)
    percentage = (np.count_nonzero(diff == 0) / diff.size) * 100

    return percentage

if __name__ == "__main__":
    # test out the evaluate function
    arr1 = np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
    arr2 = np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1 ,1, 1, 1, 1]])
    print(evaluate(arr1, arr2))