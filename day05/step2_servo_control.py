import cv2
import numpy as np
import serial  # pyserial 라이브러리 필요
import time

# 1. 아두이노 시리얼 연결 (포트 번호는 본인의 환경에 맞게 수정: 'COM4', '/dev/ttyUSB0' 등)
try:
    ser = serial.Serial('COM4', 9600, timeout=1) 
    time.sleep(2) # 연결 안정화를 위한 대기
    print("Arduino Connected!")
except Exception as e:
    print(f"Serial Connection Error: {e}")
    ser = None

# 2. 웹캠 열기 및 초기 설정
cap = cv2.VideoCapture(0)
MIN_AREA = 5000 
prev_status = False  # 이전 상태 저장 (False: 닫힘, True: 열림)

# 과제 1에서 트랙바로 찾은 최적의 HSV 값 입력
lower_color = np.array([35, 100, 100])
upper_color = np.array([85, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret: break

    # HSV 변환 및 마스크 생성
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 노이즈 제거 (모폴로지)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 면적 계산
    area = cv2.countNonZero(mask)
    
    # 현재 상태 결정 (임계값 기준)
    current_status = area > MIN_AREA

    # 3. 상태가 이전과 다를 때만 아두이노에 명령 전송
    if current_status != prev_status:
        if current_status:
            print(f"Detected! Area: {area} -> Sending 'O' (Open)")
            if ser: ser.write(b'O') # 'O' 명령 전송
            display_color = (0, 255, 0)
            status_text = "GATE OPEN"
        else:
            print(f"Lost! Area: {area} -> Sending 'C' (Close)")
            if ser: ser.write(b'C') # 'C' 명령 전송
            display_color = (0, 0, 255)
            status_text = "GATE CLOSED"
        
        # 상태 업데이트
        prev_status = current_status
    else:
        # 상태 변화가 없을 때 표시할 텍스트 설정
        if current_status:
            display_color = (0, 255, 0)
            status_text = "GATE OPEN"
        else:
            display_color = (0, 0, 255)
            status_text = "GATE CLOSED"

    # 화면 표시
    cv2.putText(frame, f"{status_text} ({area})", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
    cv2.imshow('Smart Parking System', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 리소스 해제
if ser:
    ser.close()
cap.release()
cv2.destroyAllWindows()