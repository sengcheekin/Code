# Code to test the proposed algorithm on only a single image
import skyDetector as detector
import cv2 as cv
from matplotlib import pyplot as plt 
import numpy as np
import time
from nightDetection import is_night

# Function to turn all subsequent column pixels to zero after the first zero pixel
def turn_subsequent_pixels_to_zero(image):
    h, w = image.shape
    for col in range(w):
        for row in range(h):
            if image[row, col] == 0:
                image[row:, col] = 0
                break
    return image

def detect_skyline(img):
    dilated = cv.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
    skyline = dilated - img

    return skyline

def evaluate(output, ground_truth):
    diff = cv.absdiff(output, ground_truth)
    diff = diff.astype(np.uint8)
    percentage = (np.count_nonzero(diff == 0) / diff.size) * 100

    return percentage
    
def get_coordinates_of_zero(img):
    coordinates = []
    h, w = img.shape
    for col in range(w):
        for row in range(h):
            if img[row, col] == 0:
                coordinates.append((col, row))
                break
    return np.array(coordinates)

def night_processing(img):
    kernel = np.ones((5,5),np.uint8)
    dilated = cv.dilate(img, kernel, iterations=3)
    dilated -= img
    # close
    dilated = cv.morphologyEx(dilated, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
    dilated = cv.erode(dilated, np.ones((7, 7), np.uint8), iterations=1)
    
    dilated = 255 - dilated
    dilated = cv.threshold(dilated, 127, 255, cv.THRESH_BINARY)[1]
    dilated_coordinates = get_coordinates_of_zero(dilated)

    # draw contours on new image
    new_image = np.zeros(dilated.shape, np.uint8)
    cv.drawContours(new_image, [dilated_coordinates], -1, (255, 255, 255), -1)
    new_image = 255 - new_image
    new_image = turn_subsequent_pixels_to_zero(new_image)
    new_image = cv.threshold(new_image, 127, 1, cv.THRESH_BINARY)[1]

    return new_image


if __name__ == "__main__":
    # time the program
    start = time.time()

    data_folder = "dataset/data"
    skyline_folder = "dataset/skyline"

    ### Img paths for testing. Uncomment the one you want to test
    img_path = "dataset/data/623/20120901_084151.jpg"
    # img_path = "dataset/data/684/20120603_004139.jpg"
    # img_path = "dataset/data/9730/20130101_102704.jpg"
    # img_path = "dataset/data/10917/20110809_022406.jpg"

    ### Ground truth paths for testing. Uncomment the one you want to test
    gt_path = "dataset/ground_truth/623_GT.png"
    # gt_path = "dataset/ground_truth/684_GT.png"
    # gt_path = "dataset/ground_truth/9730_GT.png"
    # gt_path = "dataset/ground_truth/10917_GT.png"

    # read and convert to binary for evaluation later
    ground_truth = cv.imread(gt_path, 0)
    ground_truth = cv.threshold(ground_truth, 127, 1, cv.THRESH_BINARY)[1]

    img = cv.imread(img_path)

    # check if it is night
    isNight = is_night(img)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if isNight:
        print("Night image detected")

        img_sky = night_processing(img)
    else:

        print("Day image detected")

        img_sky = detector.get_sky_region_gradient(img)

        img_sky = cv.morphologyEx(img_sky, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        img_sky = cv.erode (img_sky, np.ones((7, 7), np.uint8), iterations=1)
        
        img_sky = turn_subsequent_pixels_to_zero(img_sky)


    similarity = evaluate(img_sky, ground_truth)
    similarity = round(similarity, 2)
    if similarity > 90:
        print("Detection successful")
    
    # detect and save skyline. If directory does not exist, create it
    skyline = detect_skyline(img_sky)

    print(f"Time taken: {time.time() - start} seconds")

    plt.subplot(1,3,1)
    plt.imshow(ground_truth, 'gray')
    plt.title('ground truth')

    plt.subplot(1,3,2)
    plt.imshow(img_sky, 'gray')
    plt.title('Similarity: ' + str(similarity) + '%')

    plt.subplot(1,3,3)
    plt.imshow(skyline, 'gray')
    plt.title('skyline')
    plt.show()


