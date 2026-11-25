# ch17_7
from PIL import Image

pictObj = Image.new("RGB", (300, 180), "aqua")  # create aqua color Image
pictObj.save("out17_7.jpg")