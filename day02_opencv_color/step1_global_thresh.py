import cv2 as cv
import numpy as np
import urllib.request
import os

# 1. 샘플 이미지 다운로드 함수
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)

# 2. 콜백 함수 (트랙바용 빈 함수)
def nothing(x):
    pass

# 3. 이미지 읽기 및 전처리
# 'sudoku.png'를 그레이스케일로 읽기
img = get_sample("sudoku.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 노이즈 제거 (이진화 품질 향상)
img = cv.medianBlur(img, 5)

assert img is not None, "이미지를 읽을 수 없습니다."

# 4. 창 생성
cv.namedWindow('Binarization Tuner')

# 5. 트랙바 생성
# — threshold (0~255, 초기값 127)
cv.createTrackbar('Threshold', 'Binarization Tuner', 127, 255, nothing)
# — mode: 0=THRESH_BINARY, 1=THRESH_BINARY_INV
cv.createTrackbar('Mode', 'Binarization Tuner', 0, 1, nothing)

# 6. 반복문
while True:
    # 트랙바 값 읽기
    thresh_val = cv.getTrackbarPos('Threshold', 'Binarization Tuner')
    mode_val = cv.getTrackbarPos('Mode', 'Binarization Tuner')

    # 이진화 모드 결정
    # mode가 0이면 THRESH_BINARY, 1이면 THRESH_BINARY_INV
    current_mode = cv.THRESH_BINARY if mode_val == 0 else cv.THRESH_BINARY_INV

    # 이진화 적용
    _, result = cv.threshold(img, thresh_val, 255, current_mode)

    # 현재 임계값을 결과 화면에 표시
    display_res = result.copy()
    cv.putText(display_res, f'Thresh: {thresh_val}', (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 원본 | 이진화 결과 나란히 표시
    combined = np.hstack([img, display_res])

    cv.imshow('Binarization Tuner', combined)

    # 'q' 또는 ESC 키 종료
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# 7. 창 닫기 및 자원 해제
cv.destroyAllWindows()