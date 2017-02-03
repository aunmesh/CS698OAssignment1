import cv2
import numpy as np
import os
import sys
from glob import glob
import pickle

sys.path.append(os.getcwd())

#def ReWeightHist(Hist,FreqHist,Total):
#	return Hist * np.log(Total/(FreqHist+1))

#def ReWeightHist(Hist,FreqHist,Total,StopFrac=0.05):
#	Hist = Hist * np.log(Total/(FreqHist+1))
#	return Hist * (FreqHist < StopFrac * Total).astype(int)

def ReWeightHist(Hist,FreqHist,Total,UpperLimit=0.1,LowerLimit=0.02):
	temp_hist = Hist * ((FreqHist < UpperLimit*Total) * (FreqHist > LowerLimit*Total)).astype(int)
	return temp_hist.astype(np.float32)

from HistogramMatcher import *

#Function to ouput a sorted matrix(list) containing Image Folder Name, Image Name, Matching Score
#RootFolder - Rootfolder where Database Images are present
#TestImgPath - Path of the Image being Tested
#VocabPath - Path of the BoW Visual Vocabulary, an numpy array stored as np.float32
#VisualWordFreq - Frequency of the Visual Words in the Database Images
#TotalDBImages - Total Number of DataBase Images
#OutputPth - Path where the matrix has to be written
#Dextractor - Descriptor Extractor to be used
#Dmatcher - Descriptor Matcher to be used

def ScoreCalculator(RootFolder, TestImgPth, Vocab, VocabFreq, TotalDBImages, OutputPth, method = 1, UpperLimit = 0.1, LowerLimit=0.02, Dextractor = "SIFT", Dmatcher = "bf"):
	
	#Vocab = np.load(VocabPath)
	Vocab =  Vocab.astype(np.float32)
	TestImg = cv2.imread(TestImgPth)

	#Dextractor
	if(Dextractor == "SIFT"):
		Dextract = cv2.xfeatures2d.SIFT_create()

	#DMatcher
	if(Dmatcher == "bf"):
		Dmatch = cv2.BFMatcher()

	ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextract,Dmatch)
	ImgDescEx.setVocabulary(Vocab)
	Testkp = Dextract.detect(TestImg)

	TestHist = ImgDescEx.compute(TestImg,Testkp)

	#Reweighting the above calculated Histogram according to frequencies of each bin wrt to the Database
	TestHist = ReWeightHist(TestHist, VocabFreq, TotalDBImages,UpperLimit,LowerLimit)

	OutputList = [["Img_Dir_Name","DB_Img_Name","Matching_Score"]]

	subdir = glob(RootFolder + "/*/") 	#rootfolder is specified in global variables without a trailing /

	for temp in subdir:

		os.chdir(temp)
		DB_precomputed = glob(temp + "*_hist.npy*") #Loading the Histogram of DBImg

		for temp2 in DB_precomputed:

			#print("LoopStarts") 

			DBHist = np.load(temp2)
			DBHist = ReWeightHist(DBHist,VocabFreq,TotalDBImages,UpperLimit,LowerLimit)

			TempScore = HistogramMatcher(TestHist,DBHist,method)
			ImgFileName = temp2.split("/")[-1][0:-9] #9 is the length of "_hist.npy"
			
			ImgDirName = temp.split("/")[-2]
			ListElement = [ImgDirName, ImgFileName,TempScore]

			OutputList.append(ListElement)
			#print(OutputList)
			#print("LoopEnds")

		
	#OutputList = OutputList.sort(key=lambda x: x[2], reverse=True)

	return OutputList[1:]

	'''
	outfile = OutputPth + "/result"

	with open(outfile, 'wb') as fp:
	    pickle.dump(OutputList, fp)
	'''