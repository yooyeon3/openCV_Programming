import numpy as np
import cv2 as cv

def nothing(x): #콜백 함수(트랙바용 _ 빈 함수)
    pass

cap = cv.VideoCapture(0) # 웹캡 연결
cv.namedWindow('image') # 창 생성

cv.createTrackbar('H_min','image',0,179,nothing)
cv.createTrackbar('H_max','image',179,179,nothing)
cv.createTrackbar('S_min','image',50,255,nothing)
cv.createTrackbar('S_max','image',255,255,nothing)
cv.createTrackbar('V_min','image',50,255,nothing)
cv.createTrackbar('V_max','image',255,255,nothing)

while(1):
    ret, frame = cap.read() # 프레임 읽기
    if not ret:
        break

    H_min = cv.getTrackbarPos('H_min','image')
    H_max = cv.getTrackbarPos('H_max','image')
    S_min = cv.getTrackbarPos('S_min','image')
    S_max = cv.getTrackbarPos('S_max','image')
    V_min = cv.getTrackbarPos('V_min','image')
    V_max = cv.getTrackbarPos('V_max','image')

    lower_bound = np.array([H_min, S_min, V_min])
    upper_bound = np.array([H_max, S_max, V_max])

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('Original', frame)
    cv.imshow('Mask', mask)
    cv.imshow('image', res)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()