import skyDetector as detector
import cv2 as cv
from matplotlib import pyplot as plt 
import os
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

    # Dictionary to store the results
    results = {}

    folder_arr = ['623', '9730', '684']

    for img_folder in os.listdir(data_folder):
    # for img_folder in folder_arr:

        print(f"Processing {img_folder}...")
        
        successfull = 0
        numNights = 0

        # read and convert to binary for evaluation later
        ground_truth = cv.imread(f'dataset/ground_truth/{img_folder}_GT.png', 0)
        ground_truth = cv.threshold(ground_truth, 127, 1, cv.THRESH_BINARY)[1]

        for img_path in os.listdir(os.path.join(data_folder, img_folder)):

            try:            
                img = cv.imread(os.path.join(data_folder, img_folder, img_path))

                # check if it is night
                isNight = is_night(img)

                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                if isNight:
                    numNights += 1
                    img_sky = night_processing(img)
                else:

                    img_sky = detector.get_sky_region_gradient(img)

                    img_sky = cv.morphologyEx(img_sky, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    img_sky = cv.erode (img_sky, np.ones((7, 7), np.uint8), iterations=1)
                    
                    img_sky = turn_subsequent_pixels_to_zero(img_sky)


                similarity = evaluate(img_sky, ground_truth)
                if similarity > 90:
                    successfull += 1
                
                plt.subplot(2,1,1)
                plt.imshow(img_sky)
                plt.title('Similarity: ' + str(similarity) + '%')

                plt.subplot(2,1,2)
                plt.imshow(ground_truth)
                plt.title('ground truth')
                plt.show()

                # # detect and save skyline. If directory does not exist, create it
                # skyline = detect_skyline(img_sky)

                # if not os.path.exists(os.path.join(skyline_folder, img_folder)):
                #     os.makedirs(os.path.join(skyline_folder, img_folder))
                # plt.imsave(os.path.join(skyline_folder, img_folder, img_path), skyline, cmap='gray')

            except Exception as e:
                print(f"Error processing {img_folder}/{img_path}: {e}")

        success_rate = successfull / len(os.listdir(os.path.join(data_folder, img_folder)))
        success_rate = round(success_rate * 100, 2)        
        results[img_folder] = success_rate

        print(f"Success rate for {img_folder}: {success_rate}%")
        print(f"Number of night images: {numNights}, Number of day images: {len(os.listdir(os.path.join(data_folder, img_folder))) - numNights}")
        

    print(results)
    print(f"Time taken: {time.time() - start} seconds")
