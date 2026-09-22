import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector(detectionCon=0.5, trackCon=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

    if bboxInfo:
        x1, y1, w, h = bboxInfo["bbox"]
        cv2.rectangle(img, (x1, y1),
                      (x1 + w, y1 + h),
                      (255, 0, 255), 2)
        center = bboxInfo["center"]
        cv2.circle(img, center, 15, (0, 255, 255), cv2.FILLED)

    for point in lmList:
        cv2.circle(img, (point[0], point[1]), 3, (0, 255, 255), cv2.FILLED)

    if len(lmList) > 28:
        p24, p26, p28 = lmList[24][:2], lmList[26][:2], lmList[28][:2]
        angle, _ = detector.findAngle(p24, p26, p28, img)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()