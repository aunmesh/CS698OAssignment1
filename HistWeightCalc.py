import cv2
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from BoW_weightCalc import *
from globalvariables import *

subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

BoW = np.load(VocabPath)
BoW = np.astype(np.float32)

BoW_weights = np.zeros(len(BoW),np.float32)

totalimg = 0

for temp in subdir:
	totalimg = totalimg + 1
	os.chdir(temp)
	imagefiles = glob(temp + "*.jpg*")

	for l in imagefiles:
		print("loop starts") 
		
		temp = HistogramCalculator(l,BoW)
		BoW_weights = BoW_weights + (temp>0).astype(int)
		print("loop ends")

print("weights have been calculated")

np.save(rootfolder + "/BoW_weights",BoW_weights)
np.save(rootfolder + "/TotalNumberOfImages",totalimg)
