import cv2 as cv
import numpy as np

# 이미지 로드
img = cv.imread('road_image.jpg')
scale = 0.5
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 전처리: 가우시안 블러 (노이즈 제거)
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# 에지 검출
edges = cv.Canny(blurred, 100, 200, apertureSize=3)

# 모폴로지 연산: 에지 강화 (선택)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# edges = cv.dilate(edges, kernel, iterations=1)  # 선의 굵기 증가

# ROI 설정: 이미지 아래쪽만 처리 (도로 부분)
height = edges.shape[0]
roi = edges[int(height*0.5):, :]  # 이미지 아래 50% 만 사용

# Hough Line Transform
lines = cv.HoughLinesP(roi, 1, np.pi/180, threshold=50,minLineLength=30, maxLineGap=10)
# 결과 표시
result = img_resized.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # ROI 좌표를 원본 좌표로 변환
        y1_orig = y1 + int(height*0.5)
        y2_orig = y2 + int(height*0.5)
        cv.line(result, (x1, y1_orig), (x2, y2_orig), (0, 255, 0), 2)
    print(f"ROI 적용 후 검출된 직선: {len(lines)}개")

cv.imshow('Original', gray)
cv.imshow('Edges', edges)
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()
