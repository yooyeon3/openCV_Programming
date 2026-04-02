import urllib.request
import os
import numpy as np 
import cv2 as cv 

def get_sample(filename, repo='opencv'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

#img = cv.imread(get_sample('messi5.jpg', repo='opencv'))
#img = cv.imread(get_sample('messi5.jpg'), cv.IMREAD_GRAYSCALE)
#img = cv.imread(get_sample('messi5.jpg'))
img = cv.imread(get_sample('sudoku.png'))
#h, w = img.shape
h, w = img.shape[:2]
print(img.shape)

#res = cv.resize(img, None, fx=4, fy=4, interpolation = cv.INTER_CUBIC)
# 평행 이동 
#M = np.float32([[1, 0, 100], [0, 1, 50]])
#dst = cv.warpAffine(img, M, (w, h))

# 회전 이동 ( 중심점, 각도, 스케일 )
#M = cv.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), 90, 1)
#M = cv.getRotationMatrix2D(((w)/2.0, (h)/2.0), 90, 1)
#dst = cv.warpAffine(img, M, (w,h))


#3개의 점의 대응관계
#pts1 = np.float32([[50,50], [200,50], [50,200]])
#pts2 = np.float32([[10,100], [200,50], [100,250]])

#4개의 점의 대응관계
pts1 = np.float32([[0,0], [w,0], [0,h],[w,h]])
pts2 = np.float32([[0,0], [w,0], [50,0],[w-50,h]])

# 원근  변환 행렬 계산
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (w, h))

cv.imshow("Original", img)
#cv.imshow("Scaling", res)
#cv.imshow("Traslated", dst)
#cv.imshow("Rotated", dst)
#cv.imshow("Affine", dst)
cv.imshow("Perspective", dst)

cv.waitKey(0)
cv.destroyAllWindows()