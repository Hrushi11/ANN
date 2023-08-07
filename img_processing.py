import cv2
import numpy as np
from skimage import filters, morphology, segmentation
from scipy import fftpack

# Convolution using OpenCV
def apply_convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Histogram Equalization using OpenCV
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Median Filtering using OpenCV
def median_filtering(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Fourier Transform using SciPy
def fourier_transform(image):
    return fftpack.fftn(image)

# Edge detection using skimage filters
def edge_detection(image):
    return filters.sobel(image)

# Morphological operation: Dilation using skimage morphology
def dilation(image, selem):
    return morphology.dilation(image, selem)

# Otsu's Thresholding using skimage filters
def otsu_thresholding(image):
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    return binary

# Implementing SIFT is not straightforward due to patent issues.
# However, you can use the OpenCV implementation like this, if available in your version:
def sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
