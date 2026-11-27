# ch17_16
from PIL import Image

pict = Image.open("rushmore.jpg")       # create Pillow object
copyPict = pict.copy()                  # copy
copyPict.save("out17_16.jpg")