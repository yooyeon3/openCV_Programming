import cv2
import numpy as np
import os

# 1. 파일 설정
file_path = 'people2.jpg'  # 사용하실 파일명으로 수정하세요

if not os.path.exists(file_path):
    print(f"Error: '{file_path}' 파일을 찾을 수 없습니다.")
else:
    img = cv2.imread(file_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        exit()

    # 2. HOG 디스크립터 및 기본 모델 로드
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # --- Step 3-4: 파라미터 튜닝 (다양한 설정 비교) ---
    configs = [
        {"name": "Default", "winStride": (8, 8), "scale": 1.05},
        {"name": "Fine-grained", "winStride": (4, 4), "scale": 1.05},
        {"name": "Small objects", "winStride": (8, 8), "scale": 1.02},
        {"name": "Fast", "winStride": (16, 16), "scale": 1.1},
    ]

    print("\n[Step 3-4] 파라미터별 검출 결과 비교:")
    print("-" * 65)
    
    # 각 설정별로 루프를 돌며 검출 결과 출력
    for config in configs:
        found, _ = hog.detectMultiScale(
            img,
            winStride=config["winStride"],
            padding=(16, 16),
            scale=config["scale"]
        )
        print(f"{config['name']:15s} (winStride={config['winStride']}, scale={config['scale']}): {len(found):3d}명")

    print("-" * 65)
    print("✅ 분석 완료: 이미지 특성에 맞는 최적의 파라미터를 선택하세요.")


    # --- Step 3-3: 신뢰도 필터링으로 최종 검출 ---
    # 여기서는 가장 정밀한 'Small objects' 설정을 기준으로 필터링을 진행해 보겠습니다.
    BEST_WIN_STRIDE = (8, 8)
    BEST_SCALE = 1.02
    CONFIDENCE_THRESHOLD = 0.5  # 임계값 (0.3 ~ 0.9)

    detections, weights = hog.detectMultiScale(
        img,
        winStride=BEST_WIN_STRIDE,
        padding=(16, 16),
        scale=BEST_SCALE
    )

    result_filtered = img.copy()
    filtered_count = 0

    if len(detections) > 0:
        for (x, y, w, h), weight in zip(detections, weights):
            if weight > CONFIDENCE_THRESHOLD:
                # 검출 박스 시각화
                cv2.rectangle(result_filtered, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 신뢰도 점수 표시
                cv2.putText(result_filtered, f'{weight:.2f}', (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                filtered_count += 1

        # 필터링 결과 분석 출력
        print(f"\n[Step 3-3] 신뢰도 필터링 결과 (threshold={CONFIDENCE_THRESHOLD}):")
        print(f"  필터링 전: {len(detections)}명")
        print(f"  필터링 후: {filtered_count}명")
        
        if len(detections) > 0:
            reduction_rate = ((len(detections) - filtered_count) / len(detections) * 100)
            print(f"  감소율(오탐 제거): {reduction_rate:.1f}%")
    else:
        print("\n검출된 보행자가 없습니다.")

    # 5. 최종 화면 출력
    cv2.imshow('Final Filtered Detection', result_filtered)
    
    print("\n💡 아무 키나 누르면 모든 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()