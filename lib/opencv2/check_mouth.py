import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=8)

# MediaPipe FaceMesh 嘴唇關鍵點：13/14 上下內唇中點，61/291 左右嘴角
TOP_LIP, BOTTOM_LIP, LEFT_CORNER, RIGHT_CORNER = 13, 14, 61, 291

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        vertical = detector.findDistance(face[TOP_LIP], face[BOTTOM_LIP])[0]
        horizontal = detector.findDistance(face[LEFT_CORNER], face[RIGHT_CORNER])[0]
        mar = vertical / horizontal  # mouth aspect ratio，數值越大代表嘴巴張越開
        print(f"MAR: {mar:.2f} {'張嘴' if mar > 0.5 else ''}")
        

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
