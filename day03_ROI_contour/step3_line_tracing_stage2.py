import cv2 as cv
import numpy as np

# 1. 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 2', cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. 전처리 및 이진화 (검은 선을 추출하기 위해 INV 사용)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 조명 차이에 강한 Otsu 이진화 적용
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 3. [추가] 노이즈 제거 (Median Blur & Morphology)
    # 메디안 필터: 점 노이즈(Salt & Pepper) 제거에 탁월함
    binary = cv.medianBlur(binary, 5)
    
    # 모폴로지 열기: 선 주변의 자잘한 흰 점들을 지워줌
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # 4. 컨투어 검출 및 가장 큰 컨투어 찾기
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 5. 제어 정보 계산
    if largest_cnt is not None and max_area > 500:
        # (1) 중심점 계산
        M = cv.moments(largest_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # (2) [추가] 방향(각도) 계산: fitLine
            # 컨투어를 구성하는 점들에 가장 잘 맞는 직선을 찾아줍니다.
            vx, vy, x, y = cv.fitLine(largest_cnt, cv.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * 180 / np.pi # 라디안을 각도(deg)로 변환

            # (3) [추가] 제어신호(Steer) 생성: Error 값 기반
            frame_center_x = frame.shape[1] // 2
            error = cx - frame_center_x # 화면 중심에서 선이 얼마나 벗어났는가?
            steer = error / frame_center_x # -1.0(좌측 끝) ~ 1.0(우측 끝) 범위로 정규화

            # 시각화: 컨투어, 중심점, 텍스트 표시
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Angle: {angle[0]:.1f} deg', (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Steer: {steer:.2f}', (10, 90), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 6. 결과 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 2', result)

    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()