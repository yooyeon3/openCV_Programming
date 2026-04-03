import cv2
import numpy as np

# 1. 웹캠 열기 (0번은 기본 카메라)
cap = cv2.VideoCapture(0)

# 2. 감지할 색상의 HSV 범위 설정 (예: 녹색)
# 이 값은 실습 환경(조명 등)에 따라 트랙바로 조정이 필요할 수 있습니다.
lower_color = np.array([35, 100, 100])
upper_color = np.array([85, 255, 255])

# 3. 감지 면적 임계값 설정 (너무 작은 노이즈 무시)
MIN_AREA = 5000 

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # HSV 색공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 마스크 생성 (특정 색상 영역만 흰색, 나머지는 검은색)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 마스크 픽셀 면적 계산 (흰색 픽셀의 개수)
    area = cv2.countNonZero(mask)

    # 면적과 임계값 비교하여 상태 결정
    if area > MIN_AREA:
        status_text = "Status: OPEN (Detected)"
        display_color = (0, 255, 0) # 녹색
    else:
        status_text = "Status: CLOSED"
        display_color = (0, 0, 255) # 빨간색

    # 화면에 정보 표시
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
    cv2.imshow('Parking Gate System', frame)
    cv2.imshow('Mask', mask) # 필터링 결과 확인용

    # 'q' 키 입력 시 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()