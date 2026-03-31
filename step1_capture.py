import numpy as np
import cv2 as cv

frame_count = 0

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame_count = 0

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(frame)
    # Display the resulting frame
    #cv.imshow('frame', gray)
    #frame = cv.flip(frame, 0)    #기존 코드
    frame = cv.flip(frame, 1)     #좌우반전한 코드
    cv.imshow('frame',frame)
    key = cv.waitKey(1)
    # c를 누르면 캡쳐가 돼서 사진을 저장한다.

    if key == ord('c'):
        cv.imwrite("my_photo.png", frame)
        print("캡쳐 완료!")
        break # 저장 후 프로그램 종료
    
    elif key == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()