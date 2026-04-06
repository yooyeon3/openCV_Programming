import cv2 as cv
import numpy as np

# 이미지 로드
img = cv.imread('road_image.jpg')
scale = 0.5
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 다양한 Canny 임계값으로 실험
canny_params = [
    (50, 150),    # 낮은 임계값 → 더 많은 에지
    (100, 200),   # 중간 임계값 (기본값)
    (150, 250),   # 높은 임계값 → 강한 에지만
]

for lower, upper in canny_params:
    edges = cv.Canny(gray, lower, upper, apertureSize=3)
    
    # HoughLinesP 적용
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=10)
    
    result = img_resized.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"Canny({lower}, {upper}) → 검출된 직선: {len(lines)}개")
        cv.imshow(f'Canny({lower}, {upper})', result)
    else:
        print(f"Canny({lower}, {upper}) → 검출된 직선 없음")

cv.waitKey(0)
cv.destroyAllWindows()
