import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('dataset/623/20120901_084151.jpg') # night image
img = cv2.imread('dataset/623/20120901_174155.jpg') # day image
# img = cv2.imread('dataset/623/20120905_004156.jpg') # dawn image
# img = cv2.imread('dataset/9730/20130329_052708.jpg') # dusk image

alpha = 2 # Contrast control
beta = 50 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
inverse = cv2.bitwise_not(img)

# display all images
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(adjusted)
plt.title('Adjusted')

plt.subplot(1, 3, 3)
plt.imshow(inverse)
plt.title('Inverse')

plt.show()