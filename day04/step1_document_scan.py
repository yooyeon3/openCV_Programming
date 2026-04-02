import cv2 as cv
import numpy as np

# ============================================================
# 전역 변수
# ============================================================
win_name = "Document Scanning"
img = None      # 캡처된 원본 이미지
draw = None     # 화면 표시 및 점을 그릴 이미지
pts_cnt = 0     # 클릭 횟수
pts = np.zeros((4, 2), dtype=np.float32)
is_captured = False  # 현재 캡처(정지) 상태인지 여부

# ============================================================
# 마우스 콜백 함수
# ============================================================
def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts, img, is_captured
    
    # 캡처된 상태에서만 마우스 클릭을 허용
    if is_captured and event == cv.EVENT_LBUTTONDOWN:
        if pts_cnt < 4:
            # 1️⃣ 클릭한 위치에 원 표시 및 좌표 저장
            cv.circle(draw, (x, y), 10, (0, 255, 0), -1)
            cv.imshow(win_name, draw)
            pts[pts_cnt] = [x, y]
            pts_cnt += 1
            
        # 2️⃣ 4개 점 수집 완료 → 원근 변환 실행
        if pts_cnt == 4:
            # 좌표 정렬 (합과 차를 이용한 좌상/우상/우하/좌하 판단)
            sm = pts.sum(axis=1)                 # x + y
            diff = np.diff(pts, axis=1)          # y - x
            
            tl = pts[np.argmin(sm)]              # 좌상: 합이 최소
            br = pts[np.argmax(sm)]              # 우하: 합이 최대
            tr = pts[np.argmin(diff)]            # 우상: 차가 최소
            bl = pts[np.argmax(diff)]            # 좌하: 차가 최대
            
            pts_src = np.array([tl, tr, br, bl], dtype=np.float32)

            # 변환 후 출력될 서류 크기 계산 (유클리드 거리)
            w1 = np.linalg.norm(br - bl)
            w2 = np.linalg.norm(tr - tl)
            h1 = np.linalg.norm(tr - br)
            h2 = np.linalg.norm(tl - bl)
            width = int(max(w1, w2))
            height = int(max(h1, h2))

            # 변환 후 목표 좌표 (직사각형)
            pts_dst = np.array([
                [0, 0], [width - 1, 0], 
                [width - 1, height - 1], [0, height - 1]
            ], dtype=np.float32)

            # 원근 변환 행렬 계산 및 적용
            matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
            result = cv.warpPerspective(img, matrix, (width, height))

            # 결과 창 표시
            cv.imshow('Scanned Document', result)
            print("✅ 스캔 완료! 'r'을 누르면 다시 실시간 모드로 돌아갑니다.")

# ============================================================
# 메인 실행
# ============================================================
cap = cv.VideoCapture(0)
cv.namedWindow(win_name)
cv.setMouseCallback(win_name, onMouse)

print("📝 사용법:")
print("1. [Space]: 원하는 장면에서 화면을 캡처합니다.")
print("2. 마우스로 서류의 네 모서리를 클릭합니다 (순서 상관없음).")
print("3. [r]: 다시 실시간 웹캠 모드로 돌아갑니다.")
print("4. [q]: 프로그램을 종료합니다.")

while True:
    if not is_captured:
        # 실시간 모드: 카메라로부터 계속 프레임을 읽어옴
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv.resize(frame, (800, 600))
        img = frame.copy()
        draw = frame.copy()
        
        # 안내 메시지 삽입
        cv.putText(draw, "Press [Space] to Capture", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv.imshow(win_name, draw)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):    # 종료
        break
    elif key == ord(' '):  # 화면 캡처
        is_captured = True
        print("📍 화면이 고정되었습니다. 네 모서리를 클릭하세요.")
    elif key == ord('r'):  # 리셋 (실시간 모드 복귀)
        is_captured = False
        pts_cnt = 0
        print("🔄 실시간 모드로 전환합니다.")

cap.release()
cv.destroyAllWindows()