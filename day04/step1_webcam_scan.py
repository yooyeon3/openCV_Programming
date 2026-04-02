import cv2 as cv
import numpy as np

# 전역 변수
win_name = "Document Scanning"
img = None
draw = None
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
is_captured = False  # 현재 캡처 모드인지 확인

def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts, img
    
    # 캡처된 상태에서만 마우스 클릭 허용
    if is_captured and event == cv.EVENT_LBUTTONDOWN:
        cv.circle(draw, (x, y), 10, (0, 255, 0), -1)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        
        if pts_cnt == 4:
            # 좌표 정렬 로직
            sm = pts.sum(axis=1)
            tl = pts[np.argmin(sm)]
            br = pts[np.argmax(sm)]
            diff = np.diff(pts, axis=1)
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            
            pts_src = np.array([tl, tr, br, bl], dtype=np.float32)
            
            # 크기 계산
            width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
            height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

            pts_dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

            matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
            result = cv.warpPerspective(img, matrix, (width, height))

            cv.imshow('Scanned Document', result)
            pts_cnt = 0 # 초기화

# 메인 실행
cap = cv.VideoCapture(0)
cv.namedWindow(win_name)
cv.setMouseCallback(win_name, onMouse)

print("📝 사용법:")
print("- 's' 키: 현재 화면 캡처 및 점 찍기 시작")
print("- 'r' 키: 실시간 모드로 복귀 (리셋)")
print("- 'q' 키: 종료")

while True:
    if not is_captured:
        ret, frame = cap.read()
        if not ret: break
        frame = cv.resize(frame, (800, 600))
        img = frame.copy()
        draw = frame.copy()
    
    # 캡처된 상태에서는 점이 그려진 draw를 계속 보여줌
    cv.imshow(win_name, draw)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'): # 스캔 모드 진입
        is_captured = True
        print("📍 4개의 점을 클릭하세요.")
    elif key == ord('r'): # 리셋
        is_captured = False
        pts_cnt = 0
        print("🔄 실시간 모드로 돌아갑니다.")

cap.release()
cv.destroyAllWindows()