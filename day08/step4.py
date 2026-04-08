import cv2
import numpy as np
import time

class PedestrianDetectionStats:
    """보행자 감지 통계 수집"""
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.max_per_frame = 0
        self.detection_history = []
        self.start_time = time.time()
    
    def update(self, detections):
        """프레임 정보 업데이트"""
        self.frame_count += 1
        self.detection_count += len(detections)
        self.max_per_frame = max(self.max_per_frame, len(detections))
        self.detection_history.append(len(detections))
    
    def print_stats(self, title="보행자 감지 통계"):
        """통계 출력"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*50)
        print(f"📊 {title}")
        print("="*50)
        print(f"처리 시간: {elapsed:.1f}초")
        print(f"처리된 프레임: {self.frame_count}개")
        print(f"평균 FPS: {fps:.1f}")
        print(f"총 감지한 보행자: {self.detection_count}명")
        print(f"프레임당 평균: {self.detection_count/max(1, self.frame_count):.2f}명")
        print(f"최대 감지: {self.max_per_frame}명/프레임")
        
        if self.frame_count > 0:
            zero_detection = sum(1 for d in self.detection_history if d == 0)
            print(f"보행자 없는 프레임: {zero_detection}개 ({zero_detection/self.frame_count*100:.1f}%)")
        print("="*50)

# 1. HOG 디스크립터 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. 다양한 신뢰도 임계값 테스트 섹션 (PART 1)
THRESHOLDS = [0.3, 0.5, 0.7] # 빠른 테스트를 위해 3가지만 진행

print("\n" + "="*50)
print("PART 1: 신뢰도 임계값별 성능 분석 (각 100프레임)")
print("="*50)

for threshold in THRESHOLDS:
    stats_part1 = PedestrianDetectionStats() # 임계값별 개별 통계 생성
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Error: 임계값 {threshold} 테스트용 웹캠을 열 수 없습니다.")
        continue

    while stats_part1.frame_count < 100:
        ret, frame = cap.read()
        if not ret: break
        
        detections, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        # 현재 임계값에 맞는 필터링
        filtered = [(x, y, w, h) for (x, y, w, h), weight in zip(detections, weights) if weight > threshold]
        
        # 통계 업데이트
        stats_part1.update(filtered)
        
        # 시각화
        for (x, y, w, h) in filtered:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Analysis Mode - Threshold: {threshold}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {stats_part1.frame_count}/100", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Pedestrian Detection - Analysis', frame)
        if cv2.waitKey(1) == 27: break
            
    cap.release()
    cv2.destroyAllWindows()
    stats_part1.print_stats(f"Threshold {threshold} 결과 리포트")

# 3. 실시간 성능 최적화 섹션 (PART 2)
print("\n" + "="*50)
print("PART 2: 실시간 성능 최적화 모드 시작")
print("기능: 3프레임마다 검출 수행 + 통계 수집")
print("종료: 'q' 또는 ESC")
print("="*50)

stats_part2 = PedestrianDetectionStats()
PROCESS_INTERVAL = 3 
CONFIDENCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_detections = []
processing_times = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- 최적화: N프레임마다만 무거운 연산 수행 ---
    if stats_part2.frame_count % PROCESS_INTERVAL == 0:
        start_time = time.time()
        detections, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        last_detections = [
            (x, y, w, h) for (x, y, w, h), weight in zip(detections, weights) if weight > CONFIDENCE_THRESHOLD
        ]
        processing_times.append(time.time() - start_time)
    
    # 통계 업데이트 (매 프레임 호출하여 FPS 및 인원 기록)
    stats_part2.update(last_detections)
    
    # 시각화 (이전 결과 재사용)
    for (x, y, w, h) in last_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 정보 표시
    cv2.putText(frame, f"OPTIMIZED MODE (Interval: {PROCESS_INTERVAL})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(frame, f"Total Frames: {stats_part2.frame_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {len(last_detections)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Optimized Real-time Detection', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 최종 통계 출력
stats_part2.print_stats("최적화 모드 최종 결과")

if processing_times:
    final_avg_ms = np.mean(processing_times) * 1000
    print(f"💡 [기술 분석] 순수 알고리즘 처리 시간(평균): {final_avg_ms:.1f}ms")