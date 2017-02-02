import numpy as np 
import cv2

#method is given we are using the cv2 histogram matching, if we are not using that 
def HistogramMatcher(Hist1,Hist2,FreqHist,Total,method=1):
	Hist1 = Hist1.astype(np.float32)
	Hist2 = Hist2.astype(np.float32)

	Hist1 * (FreqHist < 0.002 * Total).astype(int)
	Hist2 * (FreqHist < 0.002 * Total).astype(int)

	#Hist1 = Hist1 / np.linalg.norm((Hist1), ord=1)
	#Hist2 = Hist2 / np.linalg.norm((Hist2), ord=1)

	return np.linalg.norm((Hist1 - Hist2), ord=1)

	#return cv2.compareHist(Hist1,Hist2,method)
