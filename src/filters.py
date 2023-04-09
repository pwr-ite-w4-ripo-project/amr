import numpy as np
import cv2 as cv

kernel = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
]

def test_filter(image):
    return cv.filter2D(image, -1, kernel=np.asarray(kernel))