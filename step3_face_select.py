import numpy as np
import cv2 as cv
import sys

# 1. 전역 변수 초기화
ix, iy = -1, -1
drawing = False

# 2. my_id_card.png 읽기    
img_org = cv.imread('my_id_card.png')

if img_org is None:
    print("에러: my_id_card.png 파일을 찾을 수 없습니다.")
    sys.exit()

# 화면에 계속 보여줄 작업용 이미지
img = img_org.copy()

# 3. 마우스 콜백 함수 정의
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img

    # LBUTTONDOWN: 드래그 시작, 시작점(ix, iy) 저장
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # MOUSEMOVE: 드래그 중이면
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 원본에서 img 복원 (이전 사각형 제거 효과)
            img = img_org.copy()
            # 현재 위치까지 초록색 사각형 그리기 (두께 2)
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    # LBUTTONUP: 드래그 끝
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 최종 사각형 확정
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        # 사각형 위에 "FACE" 텍스트 넣기 (시작점 ix, iy 활용)
        cv.putText(img, "FACE", (ix, iy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 4. 창 생성 + 마우스 콜백 등록
cv.namedWindow('ID Card Edit')
cv.setMouseCallback('ID Card Edit', draw_rectangle)

print("설명: 마우스로 얼굴 영역을 드래그하세요.")
print("'s': 저장 후 종료, 'q': 그냥 종료")

# 5. 반복문
while True:
    # 이미지 표시
    cv.imshow('ID Card Edit', img)
    
    # 키 입력 대기 (한 번만 호출)
    key = cv.waitKey(1) & 0xFF
    
    # 's' → my_id_card_final.png로 저장 후 break
    if key == ord('s'):
        cv.imwrite("my_id_card_final.png", img)
        print("최종 이미지 저장 완료: my_id_card_final.png")
        break
        
    # 'q' → break
    elif key == ord('q') or key == 27: # ESC도 추가
        print("저장하지 않고 종료합니다.")
        break

# 6. 창 닫기
cv.destroyAllWindows()