# ch17_12
from PIL import Image

pict = Image.open("rushmore.jpg")                               # create Pillow object
# now
pict.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save("out17_12_1.jpg")    # left right
pict.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save("out17_12_2.jpg")    # top bottom
# original
# pict.transpose(Image.FLIP_LEFT_RIGHT).save("out17_12_1.jpg")    # left right
# pict.transpose(Image.FLIP_TOP_BOTTOM).save("out17_12_2.jpg")    # top bottom