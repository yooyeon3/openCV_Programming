# 코드
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import urllib.request
import os

# github sample 주소 : https://github.com/opencv/opencv/tree/master/samples/data
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)

# 파일 읽기
#img = cv.imread("./samples/starry_night.jpg")
#img = get_sample("starry_night.jpg")
#img = get_sample("gradient.png")
#img = cv.imread('gradient.png', cv.IMREAD_GRAYSCALE)

img = get_sample("sudoku.png")
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
rest,thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
rest,thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
rest,thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
rest,thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
rest,thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]


for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()