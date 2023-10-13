# SkyRegion Identifier

A computer vision algorithm developed in Python using OpenCV to automatically identify pixels that belong to the sky region in images. The project also includes the capability to indicate whether an image is taken during daytime or nighttime and attempts to identify the skyline from the images.

## Description

This project focuses on the development of a computer vision-based system that can accurately identify the sky region in images. The algorithm is designed to handle images taken during both daytime and nighttime and provides an indication of the time of capture. Additionally, the system aims to identify the skyline within the images, facilitating a comprehensive analysis of the captured scenes.

## Dataset

The dataset utilized in this project comprises a diverse collection of daytime and nighttime images obtained from various cameras and locations, taken from the Skyfinder dataset. These images serve as the basis for training and evaluating the performance of the developed computer vision algorithm.
Original dataset can be found here: https://cs.valdosta.edu/~rpmihail/skyfinder/images/index.html

## Methodology

### Daytime Images
The method to process daytime images was adapted from the algorithm presented in [1]'s paper.

1. Modify [1]'s algorithm to process grayscale images and produce the sky region mask.
2. Perform a closing morphological operation followed by erosion to remove misclassified small pixels as the sky region.
3. Check each column for '0' pixels and set subsequent pixels to '0' to accurately identify the skyline.

### Nighttime Images

1. Apply multiple dilations with a 5x5 kernel to enhance and connect bright pixels representing the skyline.
2. Perform closing and erosion operations using appropriate kernel sizes to refine the regions of interest and reduce misclassified pixels.
3. Invert the image and convert it to binary, with '1' representing the sky region and '0' representing the non-sky region.
4. Use the coordinates of the '0' pixels to draw a contour connecting all '0' pixels to fill gaps between non-sky regions accurately.

## Sub-tasks

1. Identify whether an image is taken during daytime or nighttime.
2. Detect and delineate the skyline within the images.

## System Development

The computer vision-based system is primarily developed using the following technologies:

- Python
- OpenCV

The project harnesses the capabilities of OpenCV for image processing, analysis, and feature extraction. The implementation in Python ensures accessibility and flexibility in customizing the algorithm according to the specific requirements of the sky region identification.

## Citations
[1] Y. Shen and Q. Wang, “Sky region detection in a single image for Autonomous Ground Robot Navigation,” International Journal of Advanced Robotic Systems, vol. 10, no. 10, p. 362, 2013. doi:10.5772/56884 
