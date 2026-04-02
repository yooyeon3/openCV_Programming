import cv2 as cv
import numpy as np
import urllib.request
import os

# 💡 파일 자동 다운로드 함수 (이미지가 없을 때를 대비)
def get_sample(filename, repo='insightbook'):
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

def find_license_plate(img):
    """
    자동차 번호판을 찾는 함수
    """
    height, width = img.shape[:2]
    
    # 1️⃣ 전처리
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 히스토그램 평균화 (명암 개선: 번호판 글씨를 더 뚜렷하게)
    gray = cv.equalizeHist(gray)
    
    # 2️⃣ 에지 검출 + 모폴로지
    edges = cv.Canny(gray, 50, 150)
    
    # 가로선 강조 (번호판은 가로로 긴 직사각형이므로 가로 방향으로 팽창)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # 3️⃣ 컨투어 검출 (테두리 찾기)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # 번호판 가능 컨투어 필터링
    plate_candidates = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # 너무 작은 노이즈는 무시 (최소 면적 500)
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / h  # 가로세로 비율
            
            # 번호판의 특징: 가로가 세로보다 3~6배 정도 길다
            if 3 < aspect_ratio < 6:
                plate_candidates.append((x, y, w, h, area))
    
    # 4️⃣ 가장 큰 번호판 영역 선택 + 원근 변환
    if plate_candidates:
        # 찾은 후보들 중 면적이 가장 큰 것 1개만 선택 (진짜 번호판일 확률이 높음)
        plate_candidates.sort(key=lambda x: x[4], reverse=True)
        x, y, w, h, _ = plate_candidates[0]
        
        # 원근 변환으로 정면 시점으로 정렬해서 잘라내기
        pts = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        M = cv.getPerspectiveTransform(pts, dst_pts)
        plate = cv.warpPerspective(img, M, (w, h))
        
        return plate, (x, y, w, h)
    
    # 번호판을 못 찾았을 경우
    return None, None

# ============================================================
# 메인 실행
# ============================================================

img_name = 'my_car.jpg' 
img = cv.imread(img_name)

if img is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

# 번호판 추출 함수 실행
plate, rect = find_license_plate(img)

if plate is not None:
    x, y, w, h = rect
    
    # 원본 이미지에 검출 영역(초록색 네모) 표시
    result = img.copy()
    cv.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv.putText(result, 'License Plate', (x, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 화면에 보기 좋게 크기 조절
    plate_resized = cv.resize(plate, (200, 100))
    result_resized = cv.resize(result, (640, 480))
    
    # 결과 창 띄우기
    cv.imshow('Original with Detection', result_resized)
    cv.imshow('Extracted Plate', plate_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # 추출 결과 파일로 저장
    cv.imwrite('license_plate_extracted.png', plate)
    print("✅ 번호판 추출 완료: license_plate_extracted.png 파일이 저장되었습니다.")
else:
    print("❌ 번호판을 찾을 수 없습니다.")