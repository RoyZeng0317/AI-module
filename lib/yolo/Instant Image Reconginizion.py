import argparse, cv2, time
from ultralytics import YOLO
from detection_overlay import draw_detections
# 01. loading the model (please make sure the model name and path)
parser = argparse.ArgumentParser(description="即時物品偵測；- 降低顯示門檻，+ 提高門檻，q 離開。")
parser.add_argument("--conf", type=float, default=0.15, help="顯示門檻，預設0.15")
confidence = parser.parse_args().conf
if not 0 < confidence < 1:
    parser.error("--conf 必須介於0與1之間。")
model = YOLO(
    r"C:\Users\roy\Documents\GitHub\AI-module\tranning\object_detection_runs\602e16d7b6a548b9b322fec81c7e9b22\weights\last.pt"
)
print("使用模型：", model.ckpt_path)
print("物品類別：", "、".join(model.names.values()))

# 02. open the image resource (0 means default camera; also can use the webcam, ex: "rtsp://..." or "http://...m3u8")
source = 0
cap = cv2.VideoCapture(source)
prev_time = time.time()

# settings the camera pixel (, can improving the read speed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Can't open the camera")
    exit()

print("Start to instant recongizion... Press 'q' keypress to exit program")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't get the picture, exit to stream")
        break
    # 03. running the YOLO
    # conf: ; imgsz: ; stream=True adapt to continue pixcel image
    results = model(frame, device=0, conf=confidence, imgsz=640, verbose=False)  # 顯示門檻，不是模型準確率

    # results = model.train(
    #     data="custom_road.yaml",
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     mosaic=1.0,
    #     mixup=0.1,
    #     degrees=10.0
    # )

    # 04. draw the limit tline and types label (Ultralytics provid include draw tools)
    annotated_frame = draw_detections(frame, results[0])  # 本機繁體中文字型與可寫入的影像

    # 05. display the result
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}  CONF: {confidence:.0%} [-/+ ]", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time YOLO Detection", annotated_frame)

    # press the 'q' keypress to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key in (ord('-'), ord('_')):
        confidence = max(0.01, round(confidence - 0.05, 2))
    elif key in (ord('+'), ord('=')):
        confidence = min(0.95, round(confidence + 0.05, 2))
cap.release()
cv2.destroyAllWindows()
