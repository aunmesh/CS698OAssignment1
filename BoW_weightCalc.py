import cv2
import numpy as np

#Functio to calculate the ImageDescriptor of an Image given ImgPath and a numpy matrix(n*128)

def HistogramCalculator(Imgpath,Vocab):
	img = cv2.imread(ImgPath)
		
	#Dextractor
	SIFT = cv2.xfeatures2d.SIFT_create()

	#DMatcher
	bf = cv2.BFMatcher()

	ImgDescEx = cv2.BOWImgDescriptorExtractor(SIFT,bf)

	kp = SIFT.detect(img)
	return ImgDescEx.compute(img,kp)


