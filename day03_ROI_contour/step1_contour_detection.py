import cv2 as cv
import numpy as np

def contour_soccer_ball():
    # 1. 이미지 로드
    file_path = 'img/ball.jpg' 
    img = cv.imread(file_path)
    
    if img is None:
        print("이미지를 찾을 수 없습니다.")
        return

    # 2. HSV 색공간으로 변환
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 3. 검정색 범위 정의
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80]) 

    # 4. 마스크 생성
    mask = cv.inRange(hsv, lower_black, upper_black)

    # 5. 노이즈 제거 (모폴로지 열기)
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # 6. 컨투어 검출
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_result = img.copy()
    
    # --- 카운트 변수 초기화 ---
    total_count = len(contours) # 전체 컨투어 개수
    filtered_count = 0          # 필터링된 컨투어 개수
    # -----------------------

    for cnt in contours:
        area = cv.contourArea(cnt)
        # 너무 작은 노이즈나 너무 큰 배경 제외 (면적 필터링)
        if 500 < area < 50000:
            filtered_count += 1 # 조건 만족 시 카운트 증가
            # 검정 패턴을 빨간색 선으로 표시
            cv.drawContours(img_result, [cnt], -1, (0, 0, 255), 3)

    # 제외된 노이즈 개수 계산
    noise_count = total_count - filtered_count

    # 7. 분석 결과 출력
    print("-" * 30)
    print(f"📊 분석 결과 리포트")
    print(f"1. 전체 컨투어 개수: {total_count}")
    print(f"2. 필터링된 컨투어 개수 (축구공 패턴): {filtered_count}")
    print(f"3. 제외된 노이즈 개수: {noise_count}")
    print("-" * 30)

    # 8. 결과 출력
    cv.imshow('Original', img)
    cv.imshow('Black Mask', mask)
    cv.imshow('Result', img_result)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    contour_soccer_ball()