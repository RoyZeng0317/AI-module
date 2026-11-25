# ch17_3
from PIL import Image

rushMore = Image.open("rushmore.jpg")           # create the Pillow object
print("print the object type: ", type(rushMore))
width, height = rushMore.size                   # get the image width and height
print("width= ", width)
print("height= ", height)