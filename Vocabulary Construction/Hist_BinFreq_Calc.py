import cv2
import numpy as np
import os
from glob import glob
import sys


sys.path.append(os.getcwd())

from globalvariables import *
from Image_Hist_Calc import *

subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

BoW = np.load(VocabPath)
BoW = BoW.astype(np.float32)

BoW_weights = np.zeros(len(BoW),np.float32)

totalimg = 0

for temp in subdir:
	os.chdir(temp)
	imagefiles = glob(temp + "*.jpg*")

	for l in imagefiles:
		totalimg = totalimg + 1
		print("loop starts") 
		
		temp = HistogramCalculator(l,BoW)
		BoW_weights = BoW_weights + (temp>0).astype(int)
		print("loop ends")

print("weights have been calculated")

np.save(rootfolder + "/BoW_weights",BoW_weights)
np.save(rootfolder + "/TotalNumberOfImages",totalimg)
