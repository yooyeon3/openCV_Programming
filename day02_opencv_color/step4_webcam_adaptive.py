import cv2 as cv
import numpy as np

# 1. 콜백 함수 (트랙바용 빈 함수)
def nothing(x):
    pass

# 2. 웹캠 연결
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
    exit()

# 3. 창 생성 및 트랙바 등록
cv.namedWindow('Real-time Adaptive Tuner')

# blockSize (3~31, 초기값 11)
cv.createTrackbar('BlockSize', 'Real-time Adaptive Tuner', 11, 31, nothing)
# C (0~20, 초기값 2)
cv.createTrackbar('C', 'Real-time Adaptive Tuner', 2, 20, nothing)

# 4. 실시간 처리 반복문
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환 및 노이즈 제거 (웹캠 특성상 블러 처리가 중요함)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # 트랙바 값 읽기
    b_size = cv.getTrackbarPos('BlockSize', 'Real-time Adaptive Tuner')
    c_val = cv.getTrackbarPos('C', 'Real-time Adaptive Tuner')

    # blockSize 홀수 보장 (3 미만 방지 및 짝수 보정)
    if b_size < 3: b_size = 3
    if b_size % 2 == 0: b_size += 1

    # --- (1) Global Threshold (고정 127) ---
    _, th_global = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    cv.putText(th_global, "Global(127)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    # --- (2) Otsu 자동 이진화 ---
    _, th_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.putText(th_otsu, "Otsu Auto", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    # --- (3) Adaptive Mean ---
    th_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, b_size, c_val)
    cv.putText(th_mean, f"Mean(B:{b_size},C:{c_val})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    # --- (4) Adaptive Gaussian ---
    th_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, b_size, c_val)
    cv.putText(th_gaussian, f"Gaussian(B:{b_size},C:{c_val})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    # 2x2 결과 병합 표시
    top = np.hstack([th_global, th_otsu])
    bottom = np.hstack([th_mean, th_gaussian])
    result = np.vstack([top, bottom])

    # 화면 표시 (해상도가 너무 크면 0.5배 축소)
    display_res = cv.resize(result, (0, 0), fx=0.8, fy=0.8)
    cv.imshow('Real-time Adaptive Tuner', display_res)

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 자원 해제
cap.release()
cv.destroyAllWindows()