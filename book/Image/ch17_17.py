# ch17_17
from PIL import Image

pict = Image.open("rushmore.jpg")
copyPict = pict.copy()
cropPict = copyPict.crop((80, 30, 150, 100))
copyPict.paste(cropPic, (20, 20,))