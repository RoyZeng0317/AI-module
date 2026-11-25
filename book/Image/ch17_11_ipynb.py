# ch17_11
from PIL import Image

pict = Image.open("rushmore.jpg")                   # create Pillow object
pict.rotate(45).save("out17_11_1.jpg")              # rotate 45 degree
pict.rotate(45, expand=True).save("out17_11_2.jpg") # rotate 45 degree image fill