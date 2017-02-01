import numpy as np


#Test_Hist = ReWeightHist(TestImgHist, VisualWordFreq, TotalDBImages)

def ReWeightHist(Hist,FreqHist,Total):
	return Hist * np.log(Total/FreqHist)