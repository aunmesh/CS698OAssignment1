import numpy as np
import os
import sys
from glob import glob
import pickle

#def PreCompute(ImgPath,Vocab,FreqHist,TotImages,Dextractor="SIFT",DMatcher="bf")


sys.path.append(os.getcwd())

from Image_precompute import *
from GlobalVariables import *

subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

BoW = np.load(VocabPath)
BoW = BoW.astype(np.float32)

BoW_freq = np.load(VocabfreqPath)
BoW_freq = BoW_freq.astype(np.float32)

TotImages = np.load(TotImagesPath)

for temp in subdir:
	os.chdir(temp)
	imagefiles = glob(temp + "*.jpg*")

	for l in imagefiles:
		print("loop starts") 
		temp_kp,temp_desc,temp_hist = PreCompute(l,BoW,BoW_freq,TotImages,"SIFT","bf")		
		import pickle

		outfile_kp = l[0:-4] + "_kp"
		outfile_desc = l[0:-4] + "_desc"
		outfile_hist = l[0:-4] + "_hist"

		with open('outfile_kp', 'wb') as fp:
		    pickle.dump(temp_kp, fp)

		np.save(outfile_desc,temp_desc)
		np.save(outfile_hist,temp_hist)

		print("loop ends")

print("All the DB images have been processed for Descriptors,Keypoints and ImageHistograms")