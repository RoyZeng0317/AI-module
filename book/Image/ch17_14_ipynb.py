# ch17_14
from PIL import Image
from PIL import ImageColor

newImage = Image.new('RGBA', (300, 300), "Yellow")
for x in range(50, 251):                            # x  in range 50-250
    for y in range(50, 151):                        # y  in range 50-150
        newImage.putpixel((x, y), (0, 255, 255, 255))   # fill the green color
newImage.save("out17_14_1.png")                     # 1st to save file
for x in range(50, 251):                            # x  in range 50-250
    for y in range(151, 251):                       # y  in range 151-250
        newImage.putpixel((x, y), ImageColor.getcolor("Blue", "RGBA"))
newImage.save("out17_14_2.png")                     # 2nd to save file