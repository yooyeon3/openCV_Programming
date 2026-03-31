import cv2 as cv
import numpy as np
import os
import urllib.request

# 1. 샘플 이미지 다운로드 함수
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)

# 2. 콜백 함수 (트랙바용 빈 함수)
def nothing(x):
    pass

# 3. 이미지 읽기 (조명 불균형이 있는 sudoku.png 권장)
img = get_sample("sudoku.png")
assert img is not None, "이미지를 읽을 수 없습니다."

# 4. 창 생성
cv.namedWindow('Adaptive Thresholding Comparison')

# 5. 트랙바 생성
# — blockSize (3~31, 초기값 11)
cv.createTrackbar('BlockSize', 'Adaptive Thresholding Comparison', 11, 31, nothing)
# — C (0~20, 초기값 2)
cv.createTrackbar('C', 'Adaptive Thresholding Comparison', 2, 20, nothing)

# 6. 반복문
while True:
    # 트랙바 값 읽기
    b_size = cv.getTrackbarPos('BlockSize', 'Adaptive Thresholding Comparison')
    c_val = cv.getTrackbarPos('C', 'Adaptive Thresholding Comparison')

    # blockSize가 3 미만이면 3으로 설정, 짝수면 1 더하기 (홀수 보장)
    if b_size < 3: b_size = 3
    if b_size % 2 == 0: b_size += 1

    # --- (1) Global Threshold (고정 임계값 127) ---
    _, th_global = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    cv.putText(th_global, "Global (127)", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    # --- (2) Otsu 자동 이진화 ---
    _, th_otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.putText(th_otsu, "Otsu Auto", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    # --- (3) Adaptive Mean Threshold ---
    th_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, b_size, c_val)
    cv.putText(th_mean, f"Mean (B:{b_size}, C:{c_val})", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # --- (4) Adaptive Gaussian Threshold ---
    th_gaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, b_size, c_val)
    cv.putText(th_gaussian, f"Gaussian (B:{b_size}, C:{c_val})", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # 7. 2x2 격자로 표시
    top = np.hstack([th_global, th_otsu])
    bottom = np.hstack([th_mean, th_gaussian])
    result = np.vstack([top, bottom])

    # 결과 창의 크기가 너무 크면 조절 (선택 사항)
    display_res = cv.resize(result, (0, 0), fx=0.7, fy=0.7)
    cv.imshow('Adaptive Thresholding Comparison', display_res)

    # 'q' 또는 ESC 종료
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# 8. 창 닫기
cv.destroyAllWindows()