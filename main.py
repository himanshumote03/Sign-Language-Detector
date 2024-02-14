import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

cap.release()
