# pip install cvzone opencv
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
face_detector = FaceDetector(minDetectionCon=0.5)
hand_detector = HandDetector(detectionCon=0.5, maxHands=1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame, faces = face_detector.findFaces(frame)
    hands, frame = hand_detector.findHands(frame)
    if faces:
        print("偵測到人臉數:", len(faces))
        face = faces[0]
        print("id:", face["id"])
        print("bbox:", face["bbox"])
        print("score:", face["score"])
        print("center:", face["center"])
    if hands:
        hand1 = hands[0]
        centerPoint1 = hand1["center"]
        print(centerPoint1)
        cv2.circle(frame, centerPoint1, 10, (0, 255, 255), cv2.FILLED)
        if len(hands) == 2:
            hand2 = hands[1]
            centerPoint2 = hand2["center"]
            print(centerPoint2)
            cv2.circle(frame, centerPoint2, 10, (0, 255, 255), cv2.FILLED)
        handType1 = hand1["type"]
        print(handType1)
        cv2.putText(frame, handType1, (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()   
cv2.destroyAllWindows()