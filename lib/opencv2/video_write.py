import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
out = cv2.VideoWriter("08-11.mp4", fourcc, 20, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()