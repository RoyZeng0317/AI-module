"""使用本機微軟正黑體繪製 YOLO 框與繁體中文，不下載字型。"""
import os
from pathlib import Path

import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_detections(frame, result, empty_label='未偵測到物品：模型信心未達門檻'):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msjh.ttc"), 22)
    boxes = result.boxes
    for coords, cls, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = coords
        draw.rectangle(coords, outline="#00ff80", width=2)
        label = f"{result.names[int(cls)]} {confidence:.1%}"
        position = (max(0, x1), max(0, y1 - 28))
        draw.rectangle(draw.textbbox(position, label, font=font), fill="#102018")
        draw.text(position, label, font=font, fill="#00ff80")
    if len(boxes) == 0:
        draw.text((12, 55), empty_label, font=font, fill="#ffe080")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
