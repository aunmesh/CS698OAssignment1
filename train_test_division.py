import cv2
import numpy as np
import sys
import os
from glob import glob

sys.path.append(os.getcwd())
from globalvariables import *   	#script storing global variables such as location of root folder of image database
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

print("begin")
print(trainfolder)
for temp in subdir:
	os.chdir(temp)
	objclass = temp.split("/")[-2]
	imagefiles = glob(temp + "*.jpg*")
	
	trainpath = trainfolder + "/" + objclass
	testpath = testfolder + "/" + objclass
	
	os.system("mkdir %s" %(trainpath))
	os.system("mkdir %s" %(testpath))
	
	count=0
	totlen = len(imagefiles)
	
	for l in imagefiles:
		print("loop starts") 
		#print(l)
		if(count < 0.75 * totlen) :
			os.system("cp %s %s" %(l , trainpath))
		else :
			os.system("cp %s %s" %(l , testpath))
		
		count = count + 1
		
		#imgname = l.split(".")[-2] 
		#print(imgname)
		#np.save(imgname,descs)
		
	
