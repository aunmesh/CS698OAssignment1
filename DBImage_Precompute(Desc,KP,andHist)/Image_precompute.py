#script to compute keypoints, descriptors, and ImageHistograms using a vocabulary, and return them
import cv2
import numpy as np
import os
import sys

Dextractor = "SIFT"
DMatcher = "bf"


if(Dextractor=="SIFT"):
	Dextract = cv2.xfeatures2D.SIFT_create()

if(DMatcher=="bf")
	Dmatch = cv2.BFMatcher()

ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextract,Dmatch)

def SetVocab(Vocab):
	global ImgDescEx
	ImgDescEx.setVocabulary(Vocab)



def ReWeightHist(Hist,FreqHist,Total):
	return Hist * np.log(Total/FreqHist)

sys.path.append(os.getcwd())

def PreCompute(ImgPath, FreqHist,TotImages,Dextractor="SIFT",DMatcher="bf"):

	Img = cv2.imread(ImgPath)

	kp,desc = Dextract.detectAndCompute(Img, None)
	
	keypoint_list = []

	for point in kp:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id) 
		keypoint_list.append(temp)

	Test_Hist = ImgDescEx.compute(Img,kp)
	Test_Hist = ReWeightHist(Test_Hist,FreqHist,TotImages)

	return keypoint_list,desc,Test_Hist