import numpy as np, cv2

model_path = "models/MobileNetSSD_deploy.caffemodel"
config_path = "models/MobileNetSSD_deploy.protext.txt"
class_path = "models/MobileNetSSD_labels.txt"
class_names = []

with open(class_path, "r") as f:
    class_names = f.read().split("\n")

    net = cv2.dnn.readNet(config_path, model_path)

    img = cv2.VideoCapture(0)
while img.isOpened():
    ret, frame = img.read()
    if not ret:
        break

    
    

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)