import cv2 as cv
import numpy as np
import urllib.request
import os

def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 1. 이미지 로드 (메시 원본)
img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. 템플릿 로드 (반드시 흑백으로 읽어야 gray와 비교 가능)
# 직접 만든 'template.jpg'가 없다면 메시의 얼굴 부위를 강제로 지정해볼 수 있습니다.
template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)

if template is None:
    print("에러: 'template.jpg' 파일을 찾을 수 없습니다. 메시의 얼굴 부분을 잘라서 저장해두세요.")
else:
    # 3. Template Matching
    res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    
    # 4. 최적 위치 찾기
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc  # CCOEFF_NORMED는 높은 값이 좋은 매칭입니다.
    
    # 5. 사각형 그리기
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    result_img = img.copy()
    cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

    # 6. 결과 출력
    
    cv.imshow('Template', template)
    cv.imshow('Result', result_img)
    # Result Map은 0~1 사이의 값이라 띄울 때 밝게 보일 수 있습니다.
    cv.imshow('Result Map', res) 
    cv.waitKey(0)
    cv.destroyAllWindows()

