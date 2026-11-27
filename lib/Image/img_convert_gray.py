# pip isntall imutils
import cv2, imutils, numpy as np
from fastapi import File, UploadFile

cap = cv2.VideoCapture(0)

img = cv2.imread("example.png")
if img is None:
    raise ValueError("無法讀取 example.png，請確認檔案路徑及圖片格式")
# 轉換成灰階後再次轉回 RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
