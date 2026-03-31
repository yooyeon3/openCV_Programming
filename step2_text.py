import numpy as np
import cv2 as cv

# 1. 이미지 읽기
img = cv.imread('my_photo.png')

# 만약 이미지가 없으면 에러 방지를 위해 검은 이미지 생성
if img is None:
    print("이미지를 찾을 수 없어 기본 배경을 생성합니다.")
    img = np.zeros((512, 512, 3), np.uint8)

# 이미지 높이(h), 너비(w) 가져오기
h, w = img.shape[:2]

# --- 하단 반투명 배경 바 ---
# 1) 원본 이미지 복사
overlay = img.copy()

# 2) overlay 하단 80px 영역에 검정 사각형 채우기
# 사각형 영역: (0, h-80)부터 (w, h)까지
cv.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)

# 3) addWeighted로 img와 overlay를 50:50 합성 (0.5는 투명도)
img = cv.addWeighted(overlay, 0.5, img, 0.5, 0)

# --- 텍스트 ---
font = cv.FONT_HERSHEY_SIMPLEX

# 이름 텍스트 (조금 크게, 흰색)
# 위치 예시: 하단 바 안쪽 (x=20, y=h-45)
cv.putText(img, 'Hwang Yu Yeon', (20, h - 45), font, 1.0, (255, 255, 255), 2, cv.LINE_AA)

# 소속 텍스트 (이름 아래에 작은 크기로)
# 위치 예시: (x=20, y=h-15)
cv.putText(img, 'Yeonhee Academy', (20, h - 15), font, 0.6, (200, 200, 200), 1, cv.LINE_AA)

# 4. 결과 표시
cv.imshow("My ID Card Result", img)

# 5. 키 입력 대기
cv.waitKey(0)
cv.destroyAllWindows()

# 6. 결과 이미지 저장
cv.imwrite('my_id_card.png', img)
print("성공적으로 'my_id_card.png'를 저장했습니다.")