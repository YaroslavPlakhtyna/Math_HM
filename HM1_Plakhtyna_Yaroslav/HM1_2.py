import cv2 as cv
import urllib
import numpy as np
from google.colab.patches import cv2_imshow as cv_imshow

# Read an image
def read_image_by_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, -1)
    return img

url = "https://www.clipartmax.com/png/middle/102-1026624_logo-png-without-alpha-channel.png"
url2 = 'https://images.pond5.com/venus-atmosphere-and-without-alpha-footage-056702801_iconm.jpeg'

img = read_image_by_url(url)
img2 = read_image_by_url(url2)