import cv2 as cv
import numpy as np
from sample_download import get_sample

img = cv.imread(get_sample('sudoku.png'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# 극좌표 방식: (ρ, θ)로 직선 표현
lines = cv.HoughLines(edges, 1, np.pi/180, 200)

# 검출된 직선 그리기
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow('Hough Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()