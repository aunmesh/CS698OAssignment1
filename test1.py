import cv2
import numpy as np
import sys
import os
from glob import glob

sys.path.append(os.getcwd())
import globalvariables #script storing global variables such as location of root folder of image database
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

for temp in subdir:
	os.chdir(temp)
	imagefiles = glob(temp)
	for l in imagefiles:
		img = cv2.imread(l)
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = 500)
		(kps, descs) = sift.detectAndCompute(gray, None)
		imgname = l.split(".")[-2]
		np.save(imgname,descs)
