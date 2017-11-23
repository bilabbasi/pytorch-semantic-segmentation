from PIL import Image

im = Image.open("./example-pic.jpg")
L=list(im.getdata())
print(list(im.getdata()))