# ch17_15
from PIL import Image

pict = Image.open("rushmore.jpg")       # create Pillow object
cropPict = pict.crop((80, 30, 150, 100))    # crop area
cropPict.save("out17_15.jpg")