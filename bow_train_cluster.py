import cv2
import numpy as np
import sys
import os
from glob import glob



sys.path.append(os.getcwd())
from globalvariables import * #script storing global variables such as location of root folder of image database
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /



# Descriptor extractor class pointer...
# SIFT = cv2.xfeatures2d.SIFT_create
# dmatecher

####################### TRAINING CODE #############################

import cv2
import numpy as np

temp1 = np.load("test_desc.npy")
obs = temp1.astype(np.float32)
dictionarySize = 10000 # number of clusters
a = "./VisualWords_10k.npy" # final cluster centers

BOW = cv2.BOWKMeansTrainer(dictionarySize)
BOW.add(obs)

dictionary = BOW.cluster()
np.save(a, dictionary)

##################################################################

'''
for p in training_paths:
    image = cv2.imread(p)
    gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, dsc= sift.detectAndCompute(gray, None)
    BOW.add(dsc)
'''
'''
for temp in subdir:
	os.chdir(temp)
	descfiles = glob(temp + "*.jpg*")
	
	for l in descfiles:
		print("loop starts") 
		img = cv2.imread(l)
		gray= cv2.cvtColor(img,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		#sift = cv2.xfeatures2d.SIFT_create()
		(kps, descs) = sift.detectAndCompute(gray, None)
		BOW.add(descs)
		
		#imgname = l.split(".")[-2]
		#print(imgname)
		#np.save(imgname,descs)

#dictionary created
dictionary = BOW.cluster()
				
a = rootfolder + "/DictSome.npy"	
np.save(a, dictionary)

'''
