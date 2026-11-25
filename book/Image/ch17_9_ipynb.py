# ch17_9
from PIL import Image

pict = Image.open("rushmore.jpg")       # create Pillow object
width, height = pict.size
newPict1 = pict.resize((width*2, height))   # width is double
newPict1.save("out17_9_1.jpg")
newPict2 = pict.resize((width, height*2))   # height is double
newPict2.save("out17_9_2.jpg")