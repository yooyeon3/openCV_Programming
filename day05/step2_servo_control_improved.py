import cv2
import numpy as np
import serial
import time

# 1. 아두이노 시리얼 연결
try:
    ser = serial.Serial('COM4', 9600, timeout=1) 
    time.sleep(2)
    print("✅ Arduino Connected!")
except Exception as e:
    print(f"❌ Serial Connection Error: {e}")
    ser = None

# 2. 초기 설정
cap = cv2.VideoCapture(0)
MIN_AREA = 5000 
prev_status = False

# [개선 3] 다중 색상 범위 설정
colors = {
    'Green': {
        'lower': np.array([35, 100, 100]),
        'upper': np.array([85, 255, 255]),
        'display': (0, 255, 0)
    },
    'Blue': {
        'lower': np.array([100, 100, 100]),
        'upper': np.array([130, 255, 255]),
        'display': (255, 0, 0)
    }
}

prev_time = time.time()

while True:
    start_tick = time.time()
    
    ret, frame = cap.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    total_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    detected_any = False
    max_area = 0
    
    # --- [해결책] 글자 표시 시작 위치 설정 ---
    text_y_pos = 100 

    for name, range_val in colors.items():
        mask = cv2.inRange(hsv, range_val['lower'], range_val['upper'])
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어(윤곽선) 감지 - 단순히 countNonZero보다 객체 위치 파악에 유리
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                color_detected = True
                detected_any = True
                max_area = max(max_area, area)
                
                # [추가] 물체 주위에 사각형 그리기 (시각적 피드백 강화)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), range_val['display'], 2)

        # [수정] 해당 색상이 감지되었을 때만 텍스트 출력 및 좌표 이동
        if color_detected:
            cv2.putText(frame, f"{name} Object Detected!", (20, text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, range_val['display'], 2)
            text_y_pos += 30  # 다음 색상 글자는 30픽셀 아래에 표시하여 겹침 방지
        
        total_mask = cv2.bitwise_or(total_mask, mask)

    # 상태 판단 및 아두이노 전송 (기존 로직 유지)
    current_status = detected_any
    if current_status != prev_status:
        response_start = time.time()
        if current_status:
            print(f"🔓 OPEN - Area: {int(max_area)}")
            if ser: ser.write(b'O')
        else:
            print(" CLOSE")
            if ser: ser.write(b'C')
        
        latency = (time.time() - response_start) * 1000
        print(f"Action Latency: {latency:.2f}ms")
        prev_status = current_status

    # FPS 및 정보 표시
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    process_time = (time.time() - start_tick) * 1000

    status_text = "GATE OPEN" if current_status else "GATE CLOSED"
    color = (0, 255, 0) if current_status else (0, 0, 255)
    
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Proc: {process_time:.1f}ms", (frame.shape[1]-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Advanced Parking System', frame)
    cv2.imshow('Combined Mask', total_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if ser: ser.close()
cap.release()
cv2.destroyAllWindows()