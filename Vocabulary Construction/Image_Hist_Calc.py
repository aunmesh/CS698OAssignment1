import cv2
import numpy as np

#Function to calculate the ImageDescriptor of an Image given ImgPath and a numpy matrix(n*128)

#Dextractor is a string inputted to the function with the name of the descriptor extractor(e.g. BRIEF,SURF,SIFT)
#DMatcher is same as above
def HistogramCalculator(ImgPath,Vocab, Dext="SIFT" , DMatch="BFMatcher"):
	img = cv2.imread(ImgPath)
	
	#Dextractor
	if(Dext == "SIFT"):	
		Dextractor = cv2.xfeatures2d.SIFT_create()

	#DMatcher
	if(DMatch == "BFMatcher"):
		DMatcher = cv2.BFMatcher()

	ImgDescEx = cv2.BOWImgDescriptorExtractor(Dextractor,DMatcher)
	ImgDescEx.setVocabulary(Vocab)
	kp = Dextractor.detect(img)
	return ImgDescEx.compute(img,kp)


