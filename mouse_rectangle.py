import numpy as np
import cv2 as cv

# 전역 변수 초기화
drawing = False # 마우스가 눌려진 상태인지 확인
mode = True    # True면 직사각형, False면 곡선(원)
ix, iy = -1, -1

# 검은색 배경 이미지 생성 (함수 밖에서 미리 생성해야 함)
img = np.zeros((512, 512, 3), np.uint8)

# 마우스 콜백 함수
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                # 직사각형 그리기 (초록색)
                #cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)

                # 속이 비어있는 사각형
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            else:
                # 원 그리기 (빨간색)
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)

# 윈도우 생성 및 콜백 함수 연결
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)

while True:
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    
    # 'm'을 누르면 모드 전환 (직사각형 <-> 곡선)
    if k == ord('m'):
        mode = not mode
        print(f"현재 모드: {'직사각형' if mode else '곡선'}")
        
    # ESC를 누르면 종료
    elif k == 27:
        break

cv.destroyAllWindows()