import cv2 as cv
import numpy as np

# 1단계: 이미지 로드 및 축소
img = cv.imread('road_image.jpg')  # 인터넷에서 찾은 도로 사진
if img is None:
    print("도로 이미지를 찾을 수 없습니다")
    exit(1)

# 축소 (처리 속도 향상)
scale = 0.5
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

# 2단계: 그레이스케일 변환
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 3단계: Canny 에지 검출
edges = cv.Canny(gray, 100, 200, apertureSize=3)

# 4단계: Hough Line Transform (선분 방식)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180,
                       threshold=50, minLineLength=30, maxLineGap=10)

# 5단계: 검출된 직선을 이미지에 그리기
result = img_resized.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"검출된 직선 개수: {len(lines)}")
else:
    print("검출된 직선이 없습니다")

# 결과 표시
cv.imshow('Original', gray)
cv.imshow('Edges', edges)
cv.imshow('Hough Lines', result)
cv.waitKey(0)
cv.destroyAllWindows()

# 결과 저장
cv.imwrite('car_line_detected.jpg', result)
