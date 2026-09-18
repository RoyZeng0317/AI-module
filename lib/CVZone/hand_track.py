import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)   # 調用攝影機
detector = HandDetector(detectionCon=0.5, maxHands=1)

hands, img = detector.findHands(cap)

