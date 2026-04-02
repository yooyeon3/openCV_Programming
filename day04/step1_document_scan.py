import cv2 as cv
import numpy as np

# ============================================================
# 전역 변수
# ============================================================
win_name = "Document Scanning"
img = None
draw = None
rows, cols = 0, 0
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

# ============================================================
# 마우스 콜백 함수
# ============================================================
def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts, img
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 1️⃣ 클릭한 위치에 원 표시 및 좌표 저장
        cv.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv.imshow(win_name, draw)
        
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        
        # 2️⃣ 4개 점 수집 완료 → 좌표 정렬 + 변환
        if pts_cnt == 4:
            # 좌표 정렬 (순서: 좌상, 우상, 우하, 좌하)
            # x+y 합이 가장 작으면 좌상(tl), 가장 크면 우하(br)
            sm = pts.sum(axis=1)
            tl = pts[np.argmin(sm)]
            br = pts[np.argmax(sm)]
            
            # y-x 차가 가장 작으면 우상(tr), 가장 크면 좌하(bl)
            diff = np.diff(pts, axis=1)
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            
            # 변환 전 4개 좌표 배열 생성
            pts_src = np.array([tl, tr, br, bl], dtype=np.float32)

            # 변환 후 서류 크기 계산 (가로/세로 최대 길이)
            w1 = abs(br[0] - bl[0])
            w2 = abs(tr[0] - tl[0])
            h1 = abs(tr[1] - br[1])
            h2 = abs(tl[1] - bl[1])
            width = int(max(w1, w2))
            height = int(max(h1, h2))

            # 변환 후 4개 좌표 (직사각형)
            pts_dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # 원근 변환 행렬 계산 및 적용
            matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
            result = cv.warpPerspective(img, matrix, (width, height))

            # 결과 표시 및 초기화
            cv.imshow('Scanned Document', result)
            
            # 다시 클릭할 수 있도록 초기화 (선택사항)
            pts_cnt = 0
            draw = img.copy()
            print("스캔 완료! 다시 클릭하여 새 영역을 지정할 수 있습니다.")

# ============================================================
# 메인 실행
# ============================================================

# 이미지 로드 (경로를 본인의 환경에 맞게 수정하세요)
# 예: img = cv.imread('paper.jpg')
img = cv.imread('paper.jpg') # get_sample 함수 대신 일반 경로 사용 예시

if img is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

rows, cols = img.shape[:2]
draw = img.copy()

# 윈도우 표시 + 마우스 콜백 등록
cv.namedWindow(win_name)
cv.setMouseCallback(win_name, onMouse)

print("📝 사용법:")
print("1. 이미지 위에 4개 점을 클릭하세요 (순서 상관 없음)")
print("2. 4번째 점 클릭 후 자동으로 문서 스캔이 실행됩니다.")

cv.imshow(win_name, draw)
cv.waitKey(0)
cv.destroyAllWindows()