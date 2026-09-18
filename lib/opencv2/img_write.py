import img_to_gray

path = img_to_gray.os.path.join(img_to_gray.os.path.dirname(__file__), "koala.jpg")
img = img_to_gray.cv2.imread(path, img_to_gray.cv2.IMREAD_GRAYSCALE)
assert img is not None, "讀不到圖片，檢察路徑"

img_to_gray.cv2.imwrite("result.png", img)
img_to_gray.cv2.imshow("result.png", img)

img_to_gray.cv2.waitKey(0)
img_to_gray.cv2.destroyAllWindows()