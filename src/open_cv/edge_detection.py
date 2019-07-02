import numpy as np
import matplotlib.pyplot as plt
import cv2

image_loc = '/Users/exmachina/Work/Machine_Learning/allcodes/practical_machine_learning/src/open_cv/test_image.jpg'

def canny(image):
    # gray scaling the image
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    # Reduce noise and smoothen image using Gaussian Filter
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Calculating gradients on the image to detect steep color changes with ratio 1:3
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Return a closed region of the image we want to process
def region(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    # Filling mask with triangle
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

image = cv2.imread(image_loc)
lane_image = np.copy(image) # copying image for processing

canny = canny(lane_image)
cropped_region = region(canny)

# plt.imshow(canny)
# plt.show()
cv2.imshow('Results', cropped_region)
cv2.waitKey(0)
