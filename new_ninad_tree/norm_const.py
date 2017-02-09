import numpy as np
import cv2
import pickle

from TreeDef import *
#from Treefuncs import *


import sys
import os
from glob import glob

from globvar import *


alldict = {}
scoredict = {}
sys.path.append(os.getcwd())
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

# CHECK DEFINITION OF ROOT FOLDER AND ADJUST ACCORDINGLY


#load norm const for database images
with open(rootfolder + "Dict_NormConst_01.p", 'rb') as pfile: #load first-stage tree
    alldict = pickle.load(pfile)

with open(rootfolder + "ScoreDict_01.p", 'rb') as pfile: #load first-stage tree
    scoredict = pickle.load(pfile)


# Open the constructed tree
with open(rootfolder + "tree_10M_02.p", 'rb') as pfile: #load first-stage tree
    fTree = pickle.load(pfile)



N = 4536  #54*84

def NormConst(node,N):
	global alldict
	
	if(node.isLeaf==True or node.level == 1):				
		
		N_i = len(node.ifl)
		
		if(N_i==0):
			return
		
		else:
			for imgname in node.ifl:
				alldict[imgname] = alldict[imgname] + (node.ifl[imgname] * np.log(N/N_i))  # since all values are positive and Taking L-1 norm 
						
			#print("\n")
			return
	else:
		for childs in node.children:
			NormConst(childs,N)

NormConst(fTree,N)


#Save the dictionary

with open(rootfolder + "Dict_NormConst_01.p", 'wb') as pfile:
    pickle.dump(alldict, pfile)

with open(rootfolder + "ScoreDict_01.p", 'wb') as pfile1:
    pickle.dump(scoredict, pfile1)




