# ch17_8
from PIL import Image

pictObj = Image.new("RGBA", (300, 180))     # create a full  image
pictObj.save("out17_8.png")