# ch17_10
from PIL import Image

pict = Image.open("rushmore.jpg")       # create Pillow object
pict.rotate(90).save("out17_10_1.jpg")  # rotate 90 degree
pict.rotate(180).save("out17_10_2.jpg") # rotate 180 degree
pict.rotate(270).save("out17_10_3.jpg") # rotate 270 degree