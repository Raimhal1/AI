import cv2
import numpy as np

img = cv2.imread("kormysh.jpg")
kernel = np.ones((5,5),np.uint8)
# is used to convert an image from one color space to another
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blurred image
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
# used to detect the edges in an image.
imgCanny = cv2.Canny(img,150,200)
# used to apply the dilation operation on the given image with the specified kernel
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
# method is used to perform erosion on the image.
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
cv2.imshow("Gray Image",imgGray)
cv2.imshow("Blur Image",imgBlur)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Dialation Image",imgDialation)
cv2.imshow("Eroded Image",imgEroded)
cv2.waitKey(0)
