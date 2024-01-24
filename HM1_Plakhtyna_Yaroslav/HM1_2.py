import cv2 as cv
import urllib.request
import numpy as np
import cv2_plt_imshow as cv_imshow
#from google.colab.patches import cv2_imshow as cv_imshow

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

first_image_size = img.shape
second_image_size = img2.shape
second_image_height = int(first_image_size[0]*0.25)
second_image_width = int(first_image_size[1]*0.25)

img2_resized = img2
img2_resized = cv.resize(img2_resized, (second_image_width, second_image_height), interpolation=cv.INTER_LINEAR)

combined_image = img.copy()
row_offset = (first_image_size[0] - img2_resized.shape[0]) // 2
col_offset = (first_image_size[1] - img2_resized.shape[1]) // 2
combined_image[row_offset:row_offset+img2_resized.shape[0], col_offset:col_offset+img2_resized.shape[1]] = img2_resized

shifted_matrix = np.float32([[1, 0, 50], [0, 1, -50]])
shifted_image = cv.warpAffine(img, shifted_matrix, (img.shape[1], img.shape[0]))    

center = (img.shape[1]/2, img.shape[0]/2)
rotation_matrix = cv.getRotationMatrix2D(center, angle=42, scale=1.2)
rotated_img1 = cv.warpAffine(shifted_image, rotation_matrix, (img.shape[1], img.shape[0]))

stretch_matrix = np.float32([[1.2, 0.0, 0.0], [0.0, 1.3, 0.0]])
dsize = (int(rotated_img1.shape[1]*1.3), int(rotated_img1.shape[0]*1.2))
stretched_img1 = cv.warpAffine(rotated_img1, stretch_matrix, dsize)

shifted_matrix1 = np.float32([[1, 0, 50], [0, 1, 50]])
shifted_image1 = cv.warpAffine(stretched_img1, shifted_matrix1, (stretched_img1.shape[1], stretched_img1.shape[0]))

sheared_matrix = np.float32([[1, 0.05, 0], [0.1, 1, 0]])
sheared_img = cv.warpAffine(shifted_image1, sheared_matrix, (shifted_image1.shape[1], shifted_image1.shape[0]))

center = (sheared_img.shape[1]/2, sheared_img.shape[0]/2)
rotation_matrix2 = cv.getRotationMatrix2D(center, angle=10, scale=1)
rotated_img2 = cv.warpAffine(sheared_img, rotation_matrix2, (sheared_img.shape[1], sheared_img.shape[0]))

identity_matrix = np.eye(3)
rotation_matrix2_extended = np.concatenate((rotation_matrix2, np.array([[0, 0, 1]])))
sheared_matrix_extended = np.concatenate((sheared_matrix, np.array([[0, 0, 1]])))
shifted_matrix1_extended = np.concatenate((shifted_matrix1, np.array([[0, 0, 1]])))
stretch_matrix_extended = np.concatenate((stretch_matrix, np.array([[0, 0, 1]])))
combined_matrix = rotation_matrix2_extended.dot(sheared_matrix_extended).dot(shifted_matrix1_extended).dot(stretch_matrix_extended)
transformed_img1 = cv.warpPerspective(img, combined_matrix, (img.shape[1], img.shape[0]))

cv_imshow(img)
cv_imshow(img2_resized)
cv_imshow(combined_image)
cv_imshow(shifted_image)
cv_imshow(rotated_img1)
cv_imshow(stretched_img1)
cv_imshow(shifted_image1)
cv_imshow(sheared_img)
cv_imshow(rotated_img2)
cv_imshow(transformed_img1)