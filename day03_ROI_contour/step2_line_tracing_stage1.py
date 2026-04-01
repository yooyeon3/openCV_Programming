import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 1', cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 그레이스케일 변환 및 가우시안 블러 (노이즈 제거)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 2. 이진화 (INV 추가하여 검은 선을 흰색으로 변경)
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 3. 컨투어 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 4. 중심좌표 계산 및 시각화
    # 면적 필터를 500 정도로 높여서 작은 노이즈를 무시합니다.
    if largest_cnt is not None and max_area > 500:
        M = cv.moments(largest_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 이진화 이미지(3채널 변환) + 원본 나란히 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 1', result)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()