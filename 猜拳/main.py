"""猜拳手勢辨識遊戲：用 Teachable Machine 匯出的 TFLite 模型（models/model.tflite）判斷
玩家出的手勢，電腦隨機出拳、判勝負、累計分數。按 C 拍照判讀一回合，按 Q 離開。"""
import os
import random
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "model.tflite")
LABELS_PATH = os.path.join(HERE, "models", "labels.txt")
WINDOW = "Frame"

CHOICES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}  # key 贏 value


def load_labels(path):
    """讀 labels.txt（"0 Scissors" 這種「索引 名稱」格式），回傳 {索引: 原始標籤文字}。"""
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            idx, _, name = line.strip().partition(" ")
            if idx:
                labels[int(idx)] = name
    return labels


def normalize_choice(label):
    """labels.txt 裡的原始拼字有誤植("Papaer")，不動資料檔，改用子字串比對容錯；
    辨識不出已知三種手勢就回傳 None。"""
    low = label.lower()
    if "rock" in low:
        return "rock"
    if "sciss" in low:
        return "scissors"
    if "pap" in low:
        return "paper"
    return None


def load_interpreter():
    """用 model_content(bytes) 而非 model_path 建立 Interpreter：ai_edge_litert 底層的
    C++ CreateWrapperFromFile 在這台機器上無法正確處理含中文（"猜拳"）的路徑會直接噴
    ValueError，改成 Python 自己 open() 讀成 bytes 再餵進去即可繞過，Python 的檔案
    I/O 本身處理 Unicode 路徑沒有問題。"""
    with open(MODEL_PATH, "rb") as f:
        interpreter = Interpreter(model_content=f.read())
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def predict(interpreter, input_details, output_details, labels, frame_bgr):
    """BGR frame -> Teachable Machine 標準前處理(RGB、224x224、正規化到[-1,1]) -> 推論
    -> (原始標籤文字, 信心值)。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    x = np.expand_dims(resized.astype(np.float32) / 127.5 - 1.0, axis=0)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    best = int(np.argmax(scores))
    return labels[best], float(scores[best])


def judge(player, computer):
    """回傳 "你贏"/"電腦贏"/"平手"。"""
    if player == computer:
        return "平手"
    return "你贏" if BEATS[player] == computer else "電腦贏"


def play_round(interpreter, input_details, output_details, labels, frame, score):
    """判讀一回合、更新累計分數，回傳疊字後的結果畫面。"""
    raw_label, confidence = predict(interpreter, input_details, output_details, labels, frame)
    player = normalize_choice(raw_label)
    result = frame.copy()

    if player is None:
        lines = [f"無法辨識手勢: {raw_label}"]
    else:
        computer = random.choice(CHOICES)
        outcome = judge(player, computer)
        score[{"你贏": "你", "電腦贏": "電腦", "平手": "平手"}[outcome]] += 1
        lines = [
            f"你出: {raw_label} ({confidence:.0%})  電腦出: {computer}",
            f"結果: {outcome}",
            f"比分  你 {score['你']} : {score['電腦']} 電腦 (平手 {score['平手']})",
        ]

    for i, line in enumerate(lines):
        cv2.putText(result, line, (20, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result


def main():
    interpreter, input_details, output_details = load_interpreter()
    labels = load_labels(LABELS_PATH)
    score = {"你": 0, "電腦": 0, "平手": 0}

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cv2.imshow(WINDOW, play_round(interpreter, input_details, output_details, labels, frame, score))
            cv2.waitKey(0)  # 凍結顯示這回合結果，按任意鍵恢復即時串流

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
