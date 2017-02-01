import cv2
import numpy as np
import sys
import os
from glob import glob

sys.path.append(os.getcwd())
from globalvariables import * #script storing global variables such as location of root folder of image database
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /
total_desc = 3751258
descs_all = np.empty([total_desc+1, 128],type=np.float32)
count1 = 0
count2 = 0
total = 0

for temp in subdir:
	os.chdir(temp)
	descfiles = glob(temp + "*_desc.npy*")
	
	for l in descfiles:
		#print("loop starts") 
		#print(count)
		temp_desc = np.load(l)
		total = total + len(temp_desc)
		count2 = count2 + len(temp_desc)
		#img = cv2.imread(l)
		#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = 500)
		#(kps, descs) = sift.detectAndCompute(gray, None)
		#imgname = l.split(".")[-2]
		#print(imgname)
		#np.save(imgname,descs)
		descs_all[count1:count2] = temp_desc
		count1 = count2
		
a = rootfolder + "/DescsAll01.npy"	
np.save(a, descs_all[0:total])

