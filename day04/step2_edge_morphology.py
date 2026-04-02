import cv2 as cv
import numpy as np
import urllib.request
import os

# 1. 오류 해결을 위해 get_sample 함수를 정의합니다.
def get_sample(filename, repo='opencv'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filename)
    return filename

# ============================================================
# 메인 로직 시작
# ============================================================

# 이미지 로드
img = cv.imread(get_sample('moon_gray.jpg', repo='insightbook'))

if img is None:
    print("❌ 이미지를 불러올 수 없습니다. 인터넷 연결이나 파일명을 확인하세요.")
    exit()

# 그레이스케일 변환
if len(img.shape) < 3:
    gray = img
else:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 1. Canny 에지 검출
edges = cv.Canny(gray, 50, 150)

# 2. 모폴로지 연산
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel) # 열기: 노이즈 제거
edges_closed = cv.morphologyEx(edges_cleaned, cv.MORPH_CLOSE, kernel) # 닫기: 구멍 채우기

# 3. 시각화를 위해 병합
canny_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
cleaned_color = cv.cvtColor(edges_cleaned, cv.COLOR_GRAY2BGR)
closed_color = cv.cvtColor(edges_closed, cv.COLOR_GRAY2BGR)
img_color = img if len(img.shape) == 3 else cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# 2x2 배치
top_row = np.hstack([img_color, canny_color])
bottom_row = np.hstack([cleaned_color, closed_color])
result = np.vstack([top_row, bottom_row])

cv.imshow('Step: Original - Canny - Opening - Closing', result)
cv.waitKey(0)
cv.destroyAllWindows()