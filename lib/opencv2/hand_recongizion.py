import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
hand_detector = HandDetector(detectionCon=0.5, maxHands=1)
