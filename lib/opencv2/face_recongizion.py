# pip install cvzone opencv
import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame, faces = detector.findFaces(frame)
    if faces:
        print("偵測到人臉數:", len(faces))
        face = faces[0]
        print("id:", face["id"])
        print("bbox:", face["bbox"])
        print("score:", face["score"])
        print("center:", face["center"])

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()