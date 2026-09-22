import numpy as np, cv2, os

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model_path = os.path.join(d, "MobileNetSSD_deploy.caffemodel")
config_path = os.path.join(d, "MobileNetSSD_deploy.prototxt.txt")  # 原本寫成 protext，Caffe 設定檔副檔名是 prototxt
class_path = os.path.join(d, "MobileNetSSD_labels.txt")

for p in (model_path, config_path, class_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到模型檔: {p}")

with open(class_path, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

net = cv2.dnn.readNet(config_path, model_path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = np.squeeze(net.forward())  # shape: (偵測數, 7)

    for det in detections:
        confidence = det[2]
        if confidence > 0.5:
            idx = int(det[1])
            startX, startY, endX, endY = (det[3:7] * np.array([w, h, w, h])).astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (10, 255, 0), 2)
            label = f"{class_names[idx] if idx < len(class_names) else idx}: {confidence * 100:.2f}%"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q 或 ESC 離開
        break

cap.release()
cv2.destroyAllWindows()
