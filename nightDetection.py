# Find brightness of image and compare it to a threshold value
# If the brightness is below the threshold, it is night
# If the brightness is above the threshold, it is day

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def is_night(img):
    # Find brightness of top half of image
    h, w, _ = img.shape
    img = img[0:int(h/2), 0:w]
    brightness = np.mean(img)

    # Compare brightness to threshold
    if brightness < 129:
        isNight = 1
    else:
        isNight = 0

    return isNight


if __name__ == "__main__":

    # dictionary to store ground truths of first 150 images from all 4 datasets.
    ground_truth = {
        # 0: "day", 1: "night" 
        "623": [1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0],
        "684": [0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
        "9730": [0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1],
        "10917": [1,1,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0]
    }


    folder = "dataset/data"
    results = {}

    # Iterate through all folders in dataset
    for folder_path in os.listdir(folder):

        img_paths = os.listdir(os.path.join(folder, folder_path))
        gt = ground_truth.get(folder_path)
        success = 0
        print("Processing folder: " + folder_path)

        for i in range(150):
            img = cv2.imread(os.path.join(folder, folder_path, img_paths[i]))

            isNight = is_night(img)

            # Compare to ground truth
            if isNight == gt[i]:
                success += 1
            
        results[folder_path] = round(success/150, 2)

    print(results)
        
            