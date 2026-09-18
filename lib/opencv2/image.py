# pip install opencv-python==4.5.4.60
import cv2, os

path = os.path.join(os.path.dirname(__file__), "penguins.jpg")
img = cv2.imread(path)
assert img is not None, "讀不到圖片，檢查路徑"

h, w, c = img.shape
print("影像高:", h)
print("影像寬:", w)

resized_img = cv2.resize(img, (400, 300))
print(resized_img.shape)

cv2.imshow("Penguins", img)
cv2.imshow("Penguins:resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()