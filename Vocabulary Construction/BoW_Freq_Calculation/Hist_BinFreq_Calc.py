import cv2
import numpy as np
import os
from glob import glob
import sys
import time

sys.path.append(os.getcwd())
from GlobalVariables import *

BoW = np.load(VocabPath)
BoW = BoW.astype(np.float32)

BoW_weights = np.zeros(len(BoW))


Dext = "SIFT"
DMatch = "BFMatcher"

#Dextractor
if(Dext == "SIFT"):	
	Dextractor = cv2.xfeatures2d.SIFT_create()

#DMatcher
if(DMatch == "BFMatcher"):
	DMatcher = cv2.BFMatcher()

ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextractor,DMatcher)
ImgDescEx.setVocabulary(BoW)


#Function to calculate the ImageDescriptor of an Image given ImgPath and a numpy matrix(n*128)
#Dextractor is a string inputted to the function with the name of the descriptor extractor(e.g. BRIEF,SURF,SIFT)
#DMatcher is same as above

def HistogramCalculator(ImgPath):
	img = cv2.imread(ImgPath)
	
	kp = Dextractor.detect(img)
	result = ImgDescEx.compute(img,kp)
	return result



subdir = glob(RootFolder + "/*/") #rootfolder is specified in global variables without a trailing /

totalimg = 0

count1=0
count2 = 0

start_time = time.time()

for temp in subdir:
	print("In " + str(count1+1) + " directory out of " + str(len(subdir)))
	os.chdir(temp)
	count1+=1
	imagefiles = glob(temp + "*.jpg*")
	count2 = 0
	for l in imagefiles:
		totalimg = totalimg + 1
		if(count2%5==0):
			print("In " + str(count2+1) + " image out of " + str(len(imagefiles)))
		count2+=1
		temp2 = HistogramCalculator(l)
		BoW_weights = BoW_weights + (temp2>0).astype(int)
	print("Dir took %d seconds" %(time.time() - start_time))
	start_time = time.time()

print("weights have been calculated")

np.save(RootFolder + "/BoW_weights",BoW_weights)
np.save(RootFolder + "/TotalNumberOfImages",totalimg)