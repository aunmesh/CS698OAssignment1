import cv2
import numpy as np

img = cv2.imread('N2_330.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = 500)
(kps, descs) = sift.detectAndCompute(gray, None)

#sift = cv2.SIFT()
#kp = sift.detect(gray,None)

cv2.drawKeypoints(gray,kps,img)

cv2.imwrite('sift_keypoints_01.jpg',img)
