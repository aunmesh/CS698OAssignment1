#script to compute keypoints, descriptors, and ImageHistograms using a vocabulary, and return them
import cv2
import numpy as np
import os
import sys

def ReWeightHist(Hist,FreqHist,Total):
	return Hist * np.log(Total/FreqHist)

sys.path.append(os.getcwd())

def PreCompute(ImgPath,VocabPath,FreqHist,TotImages,Dextractor="SIFT",DMatcher="bf"):

	Img = cv2.imread(ImgPath)

	if(Dextractor=="SIFT"):
		Dextract = cv2.xfeatures2D.SIFT_create()


	if(DMatcher=="bf")
		Dmatch = cv2.BFMatcher()

	kp,desc = Dextract.detectAndCompute(Img, None)
	
	keypoint_list = []

	for point in kp:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id) 
		keypoint_list.append(temp)

	Vocab = np.load(VocabPath)
	Vocab =  Vocab.astype(np.float32)

	ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextract,Dmatch)
	ImgDescEx.setVocabulary(Vocab)

	Test_Hist = ImgDescEx.compute(Img,kp)
	Test_Hist = ReWeightHist(Test_Hist,FreqHist,TotImages)

	return keypoint_list,desc,Test_Hist