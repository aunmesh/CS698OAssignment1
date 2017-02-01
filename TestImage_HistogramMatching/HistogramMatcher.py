import numpy as np 
import cv2

#method is given we are using the cv2 histogram matching, if we are not using that 
def HistogramMatcher(Hist1,Hist2,method=1):
	Hist1 = Hist1.astype(np.float32)
	Hist2 = Hist2.astype(np.float32)
	return cv2.compareHist(Hist1,Hist2,method)
