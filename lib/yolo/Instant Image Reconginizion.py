import cv2, time
from ultralytics import YOLO
# 01. loading the model (please make sure the model name and path)
model = YOLO("yolo26n.pt")

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
    results = model(frame, conf=0.15, imgsz=640, verbose=False)
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
    annotated_frame = results[0].plot()

    # 05. display the result
    cv2.imshow("Real-Time YOLO Detection", annotated_frame)

    # press the 'q' keypress to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cap.release()
cv2.destroyAllWindows()