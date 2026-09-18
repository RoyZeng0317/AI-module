# pip install imutils
import imutils, cv2, os

path = os.path.join(os.path.dirname(__file__), "koala.jpg")
img = cv2.imread(path)
assert img is not None, "讀不到圖片，檢查路徑"
resized_img = imutils.resize(img, width=200, height=200)

cv2.imshow("Koala", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()