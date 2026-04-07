import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# 시각화를 위한 함수 정의 (Step 7에서 호출)
def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    res = cv.drawMatches(img1, kp1, img2, kp2, matches, None, 
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(res)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ========== Step 1: 이미지 로드 ==========
img1 = cv.imread(get_sample('box.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(get_sample('box_in_scene.png'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

# ========== Step 2: 특징점 검출기 초기화 ==========
# TODO: SIFT 생성
sift = cv.SIFT_create() 
# 만약 ORB를 쓴다면: sift = cv.ORB_create()

# ========== Step 3: 키포인트와 디스크립터 추출 ==========
# TODO: detectAndCompute 구현
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========
# TODO: SIFT(Float 기반)에 맞는 KDTREE 설정
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# 만약 ORB(Binary 기반)를 쓴다면 아래 설정 사용:
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)

flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========
# TODO: knnMatch 구현 (가장 유사한 점 2개 추출)
matches = flann.knnMatch(des1, des2, k=2)

print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 (Ratio Test) ==========
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        # TODO: 첫 번째 매칭(m)이 두 번째 매칭(n)보다 충분히 가까운지 확인 (0.7 비율)
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

# ========== Step 7: 시각화 ==========
if len(good_matches) >= 10:
    draw_matches(img1, kp1, img2, kp2, good_matches,
                 title=f"Good Matches ({len(good_matches)})")
else:
    print(f"Not enough good matches! (Found: {len(good_matches)})")