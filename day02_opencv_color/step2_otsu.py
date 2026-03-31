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

# 3. 이미지 읽기 (그레이스케일)
img = get_sample("sudoku.png")
# 노이즈가 있으면 Otsu 성능이 떨어지므로 가우시안 블러 적용
img_blur = cv.GaussianBlur(img, (5, 5), 0)

assert img is not None, "이미지를 읽을 수 없습니다."

# 4. 창 생성
cv.namedWindow('Otsu Comparison')

# 5. 트랙바 생성
# — manual_thresh (0~255, 초기값 127): 수동 이진화용
cv.createTrackbar('Manual_Th', 'Otsu Comparison', 127, 255, nothing)
# — mode: 0=원본|수동|Otsu 비교, 1=수동|Otsu만 크게 비교
cv.createTrackbar('Mode', 'Otsu Comparison', 0, 1, nothing)

# 6. 반복문
while True:
    # 트랙바 값 읽기
    manual_val = cv.getTrackbarPos('Manual_Th', 'Otsu Comparison')
    mode_val = cv.getTrackbarPos('Mode', 'Otsu Comparison')

    # --- 수동 이진화 ---
    _, thresh_manual = cv.threshold(img_blur, manual_val, 255, cv.THRESH_BINARY)
    cv.putText(thresh_manual, f'Manual: {manual_val}', (15, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # --- Otsu 자동 이진화 ---
    # 임계값을 0으로 설정하면 Otsu 알고리즘이 최적값을 계산하여 ret_otsu에 반환함
    ret_otsu, thresh_otsu = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.putText(thresh_otsu, f'Otsu: {ret_otsu:.0f}', (15, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 7. 비교 표시 (Mode에 따른 분기)
    if mode_val == 0:
        # 원본 | 수동 | Otsu 세 개 나란히 표시
        res = np.hstack([img, thresh_manual, thresh_otsu])
    else:
        # 수동 | Otsu 두 개만 크게 표시
        res = np.hstack([thresh_manual, thresh_otsu])

    cv.imshow('Otsu Comparison', res)

    # 'q' 또는 ESC 키 종료
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# 8. 창 닫기
cv.destroyAllWindows()