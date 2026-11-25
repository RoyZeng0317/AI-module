# ch17_5
from PIL import Image

rushMore = Image.open("rushmore.jpg")   # create the Pillow object
print("print the object format: ", rushMore.format)
print("print the object description: ", rushMore.format_description)