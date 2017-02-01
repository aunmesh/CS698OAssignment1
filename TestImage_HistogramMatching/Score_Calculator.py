import cv2
import numpy as np
import os
import sys
from glob import glob

sys.path.append(os.getcwd())

def ReWeightHist(Hist,FreqHist,Total):
	return Hist * np.log(Total/FreqHist)

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

def ScoreCalculator(RootFolder, TestImgPth, VocabPath, VisualWordFreq, TotalDBImages, OutputPth , Dextractor = "SIFT", Dmatcher = "bf"):
	
	Vocab = np.load(VocabPath)
	Vocab =  Vocab.astype(np.float32)
	Test_Img = cv2.imread(TestImgPth)

	#Dextractor
	if(Dextractor == "SIFT"):
		Dextract = cv2.xfeatures2d.SIFT_create()

	#DMatcher
	if(Dmatcher == "bf"):
		Dmatch = cv2.BFMatcher()

	ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextract,Dmatch)
	ImgDescEx.setVocabulary(Vocab)
	Test_kp = SIFT.detect(TestImg)

	Test_Hist = ImgDescEx.compute(TestImg,Test_kp)

	#Reweighting the above calculated Histogram according to frequencies of each bin wrt to the Database
	Test_Hist = ReWeightHist(TestImgHist, VisualWordFreq, TotalDBImages)

	Output_List = [["Img_Dir_Name","DB_Img_Name","Matching_Score"]]

	subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

	for temp in subdir:

		os.chdir(temp)
		DB_precomputed = glob(temp + "*_hist.npy*") #Loading the Histogram of DBImg

		for temp2 in DB_precomputed:

			print("LoopStarts") 

			DB_Hist = np.load(temp2)
			Temp_Score = HistogramMatcher(Test_Hist,DB_Hist)
			ImgFileName = temp2.split("/")[-2][0:9] #9 is the length of "_hist.npy"
			ImgDirName = temp.split("/")[-2]
			ListElement = [ImgDirName, ImgFileName,Temp_Score]

			Output_List.append(ListElement)
			print("LoopEnds")

	return Output_List.sort(key=lambda x: x[2], reverse=True)