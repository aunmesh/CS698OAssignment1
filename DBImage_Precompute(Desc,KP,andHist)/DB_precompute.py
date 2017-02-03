import numpy as np
import os
import sys
from glob import glob
import pickle
import cv2
import time


sys.path.append(os.getcwd())
#from Image_precompute import *
from GlobalVariables import *


#############################################################################

Dextractor = "SIFT"
DMatcher = "bf"


BoW = np.load(VocabPath)   #VocabPaath is specified in globalvariables
BoW = BoW.astype(np.float32)

BoW_freq = np.load(VocabFreqPath) #Specified in globalvariables
BoW_freq = BoW_freq.astype(np.float32)

TotImages = np.load(TotImagesPath)


if(Dextractor=="SIFT"):
	Dextract = cv2.xfeatures2d.SIFT_create()

if(DMatcher=="bf"):
	Dmatch = cv2.BFMatcher()


ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextract,Dmatch)
ImgDescEx.setVocabulary(BoW)


def ReWeightHist(Hist,FreqHist,Total):
	return Hist * np.log(Total/(FreqHist+1))

def PreCompute(ImgPath):
	Img = cv2.imread(ImgPath)

	kp,desc = Dextract.detectAndCompute(Img, None)
	keypoint_list = []

	for point in kp:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id) 
		keypoint_list.append(temp)

	Test_Hist = ImgDescEx.compute(Img,kp)
	#Test_Hist = ReWeightHist(Test_Hist,BoW_freq,TotImages)

	return keypoint_list,desc,Test_Hist
	#return keypoint_list

#############################################################################


#This Section uses the above code to calculate for whole DB, trainnig set

subdir = glob(RootFolder + "/*/") #rootfolder is specified in global variables without a trailing /

count1 = 0
count2 = 0

start_time = time.time()

for temp in subdir:

	os.chdir(temp)
	imagefiles = glob(temp + "*.jpg*")
	#imagefiles = glob(temp + "*_hist.npy*")
	print("In " + str(count1+1) + " directory out of " + str(len(subdir)))
	count1+=1

	for l in imagefiles:
		if(count2%5==0):
			print("In " + str(count2+1) + " image out of " + str(len(imagefiles)))
		count2+=1
 
 		
		temp_kp,temp_desc,temp_hist = PreCompute(l)

		outfile_kp = l[0:-4] + "_kp"
		outfile_desc = l[0:-4] + "_desc"
		outfile_hist = l[0:-4] + "_hist"

		#Pickle is used for saving list, which is temp_kp
		with open(outfile_kp, 'wb') as fp:
		    pickle.dump(temp_kp, fp)
		

		np.save(outfile_desc,temp_desc)
		np.save(outfile_hist,temp_hist)
		
	print("Dir took %d seconds" %(time.time() - start_time))
	start_time = time.time()



print("All the DB images have been processed for Descriptors,Keypoints and ImageHistograms")