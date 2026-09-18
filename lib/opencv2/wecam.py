import cv2

cap = cv2.VideoCapture(0)
# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 800,600)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)   # 畫面尺寸
cap.set(cv2.CAP_PROP_FPS, 25)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()