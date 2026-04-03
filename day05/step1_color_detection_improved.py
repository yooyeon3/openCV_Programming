import cv2
import numpy as np

# 트랙바 콜백 함수 (필수)
def nothing(x):
    pass

# 1. 윈도우 생성 및 트랙바 부착
cv2.namedWindow('Parking Gate System')

# 초기값 설정 (기존에 설정하셨던 녹색 범위를 초기값으로 입력)
cv2.createTrackbar('L_H', 'Parking Gate System', 35, 179, nothing)
cv2.createTrackbar('L_S', 'Parking Gate System', 100, 255, nothing)
cv2.createTrackbar('L_V', 'Parking Gate System', 100, 255, nothing)
cv2.createTrackbar('U_H', 'Parking Gate System', 85, 179, nothing)
cv2.createTrackbar('U_S', 'Parking Gate System', 255, 255, nothing)
cv2.createTrackbar('U_V', 'Parking Gate System', 255, 255, nothing)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 감지 면적 임계값
MIN_AREA = 5000 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. 트랙바에서 실시간으로 값 읽어오기
    l_h = cv2.getTrackbarPos('L_H', 'Parking Gate System')
    l_s = cv2.getTrackbarPos('L_S', 'Parking Gate System')
    l_v = cv2.getTrackbarPos('L_V', 'Parking Gate System')
    u_h = cv2.getTrackbarPos('U_H', 'Parking Gate System')
    u_s = cv2.getTrackbarPos('U_S', 'Parking Gate System')
    u_v = cv2.getTrackbarPos('U_V', 'Parking Gate System')

    # 읽어온 값으로 범위 업데이트
    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

    # HSV 변환 및 마스크 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 노이즈 제거 
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 작은 점들 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 객체 내부 구멍 채우기

    # 면적 계산 및 상태 결정
    area = cv2.countNonZero(mask)
    if area > MIN_AREA:
        status_text = f"Status: OPEN ({area})"
        display_color = (0, 255, 0)
    else:
        status_text = f"Status: CLOSED ({area})"
        display_color = (0, 0, 255)

    # 결과 화면 출력
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
    
    # 원본 화면과 마스크 화면을 보여줌
    cv2.imshow('Parking Gate System', frame)
    cv2.imshow('Mask Filter', mask) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 종료 시 현재 설정된 최종 값을 출력 (나중에 코드 고정용)
        print(f"Final HSV Range: lower={lower_color}, upper={upper_color}")
        break

cap.release()
cv2.destroyAllWindows()