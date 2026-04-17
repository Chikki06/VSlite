from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image size
im = Image.open("output/20240528-113345/20240528-113345.tif")
print(im.size)