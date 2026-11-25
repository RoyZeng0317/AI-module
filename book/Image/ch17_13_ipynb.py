# ch17_13
from PIL import Image

newImage = Image.new('RGBA', (300, 180), "Yellow")
print(newImage.getpixel((150, 50)))     # print the center of color
newImage.save("out17_13.png")