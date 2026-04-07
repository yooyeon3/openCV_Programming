# ========== 실습 1 코드 이후에 계속 ==========
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
img1 = cv.imread(get_sample('book2.jpg'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(get_sample('book_in_scene.png'), cv.IMREAD_GRAYSCALE)

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
# ========== Step 1: 호모그래피 계산 ==========
MIN_MATCH_COUNT = 10

if len(good_matches) >= MIN_MATCH_COUNT:
    # 1) good_matches에서 키포인트 좌표 추출 및 배열 변환
    # queryIdx는 img1(원본), trainIdx는 img2(배경)의 인덱스입니다.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 2) 호모그래피 행렬 계산 (RANSAC 알고리즘 사용)
    # RANSAC은 잘못된 매칭(Outlier)을 걸러내고 가장 믿을만한 변환 행렬 M을 찾습니다.
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    if M is not None:
        # 3) 원본 이미지(img1)의 네 모서리 좌표 정의
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # 4) perspectiveTransform으로 배경 이미지(img2)에서의 좌표 계산
        dst = cv.perspectiveTransform(pts, M)
        
        # ========== Step 2: 결과 시각화 (폴리곤 그리기) ==========
        # img2가 그레이스케일이라면 컬러로 변환하여 파란 선이 잘 보이게 합니다.
        result_img = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        
        # cv.polylines로 변환된 좌표(dst)를 따라 사각형 그리기
        # [np.int32(dst)]는 좌표를 정수형으로 변환하여 리스트로 전달합니다.
        result_img = cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
        plt.title('Detected Object with Homography (Blue Box)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # ========== Step 3: 매칭 시각화 (inlier만) ==========
        matchesMask = mask.ravel().tolist() # RANSAC이 선택한 정상 매칭점 마스크
        
        draw_params = dict(
            matchColor=(0, 255, 0),       # 정상 매칭(Inlier)은 초록색
            singlePointColor=None,
            matchesMask=matchesMask,      # 마스크를 적용해 Inlier만 표시
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Inlier만 선으로 연결하여 시각화
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img3)
        plt.title(f'Inlier Matches Only (Total Inliers: {sum(matchesMask)})')
        plt.axis('off')
        plt.show()

        inlier_count = sum(matchesMask)
        outlier_count = len(matchesMask) - inlier_count
        print(f"Inliers: {inlier_count}, Outliers: {outlier_count}")
        
    else:
        print("Failed to compute homography")
else:
    print(f"Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT})")