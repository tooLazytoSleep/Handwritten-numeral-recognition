import numpy as np
from keras.models import load_model
import os
import cv2
#Load model
model = load_model("myModel.h5")

#Load data
predict_dir = 'data'
files = os.listdir(predict_dir)
print("The data fold includes: ", files)
origin_img = []
images = []
for file in os.listdir(predict_dir):
    imgPath = os.path.join(predict_dir, file)
    image = cv2.imread(os.path.expanduser(imgPath))
    origin_img.append(image)
    x = cv2.resize(image, (28, 28))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    (thresh, binary) = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
    dst = np.zeros((28, 28, 1), np.uint8)
    for i in range(28):
        for j in range(28):
            grayPixel = 255 - binary[i, j]
            dst[i, j] = grayPixel
    images.append(dst)
    img = np.array(images)

co_orimg = origin_img[0]

for i in range(len(origin_img)-1):
    co_orimg = np.concatenate([co_orimg, origin_img[i+1]], axis=1)

co_binimg = images[0]
for i in range(len(images)-1):
    co_binimg = np.concatenate([co_binimg, images[i+1]], axis=1)

#Predict data and output result
pre_y = model.predict(img)
print("After classify:")
for i in range(len(pre_y)):
    if pre_y[i][0] == 1:
        print(files[i],"digit is odd")
    elif pre_y[i][1] == 1:
        print(files[i],"digit is even")
cv2.imshow("original_img(1~9)", co_orimg)
cv2.imshow("binary_img(1~9)", co_binimg)
cv2.waitKey(0)