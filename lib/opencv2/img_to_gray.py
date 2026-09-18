# 灰階圖像
import cv2, os

path = os.path.join(os.path.dirname(__file__), "koala.jpg")
gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
assert gray_img is not None, "讀不到圖片，檢查路徑"

cv2.imshow("Koala:gray", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()