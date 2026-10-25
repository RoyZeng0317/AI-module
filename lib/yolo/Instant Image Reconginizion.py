import argparse
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

# 專案根目錄（lib/yolo/ 往上兩層），讓不管從哪個資料夾執行都找得到權重與 data.yaml
ROOT = Path(__file__).resolve().parents[2]

# 預設用自己訓練的權重（circle / square / triangle），標籤以訓練時的 data.yaml 為準
DEFAULT_WEIGHTS = ROOT / "tranning" / "yolo_runs" / "custom" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "tranning" / "yolo_data" / "data.yaml"
FALLBACK_WEIGHTS = ROOT / "yolo26n.pt"  # COCO 預訓練（只認得 person/car 等 80 類）


def load_label_names(data_yaml):
    """從 data.yaml 讀出 {class_id: name}；檔案不存在就回傳 None。"""
    path = Path(data_yaml)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in (names or {}).items()} or None


def resolve_labels(model_names, data_names):
    """以 data.yaml 的標籤為準；類別數對不上就保留模型內建標籤並提醒。"""
    model_names = dict(model_names)
    if data_names is None:
        return model_names
    if len(data_names) != len(model_names):
        print(f"[警告] data.yaml 有 {len(data_names)} 類，但模型有 {len(model_names)} 類，"
              "這兩個不是同一組訓練結果，改用模型內建標籤")
        return model_names
    if data_names != model_names:
        print(f"[提醒] 模型內建標籤 {model_names} 與 data.yaml 不同，改用 data.yaml：{data_names}")
    return data_names


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 即時影像辨識")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型權重 .pt")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="訓練用 data.yaml（讀標籤名稱）")
    p.add_argument("--source", default="0", help="0=預設攝影機；也可以是影片/圖片路徑或 rtsp/http 串流")
    p.add_argument("--conf", type=float, default=0.5, help="信心門檻，太低會出現錯誤標籤")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()

    # 01. 載入模型
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[警告] 找不到 {weights}，改用 COCO 預訓練 {FALLBACK_WEIGHTS.name}")
        weights = FALLBACK_WEIGHTS
        args.data = None  # COCO 權重不對應自訓練的 data.yaml
    model = YOLO(str(weights))

    labels = resolve_labels(model.names, load_label_names(args.data) if args.data else None)
    print(f"使用權重：{weights}")
    print(f"標籤：{labels}")

    # 02. 開啟影像來源（數字代表攝影機編號）
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Can't open the camera / source")
        return

    # 設定攝影機解析度（加快讀取速度）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Start to instant recongizion... Press 'q' keypress to exit program")
    prev_time = time.time()
    last_seen = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't get the picture, exit to stream")
            break

        # 03. 執行 YOLO
        result = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        result.names = labels  # 讓畫框文字用正確的標籤

        # 終端機印出偵測到的標籤（有變化才印，避免洗版）
        seen = sorted(f"{labels[int(c)]} {float(s):.2f}"
                      for c, s in zip(result.boxes.cls, result.boxes.conf))
        if seen != last_seen:
            print("偵測到：", ", ".join(seen) if seen else "（無）")
            last_seen = seen

        # 04. 畫框與標籤
        annotated_frame = result.plot()

        # FPS 要在顯示之前畫上去才看得到
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 05. 顯示結果
        cv2.imshow("Real-Time YOLO Detection", annotated_frame)

        # 按 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
