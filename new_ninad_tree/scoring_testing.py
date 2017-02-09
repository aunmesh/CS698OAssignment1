import numpy as np
import cv2
import pickle

import sys
import os
from glob import glob

from globvar import * #script storing global variables such as location of root folder of image database

from Treefuncs import *

with open(rootfolder + "tree_10M_02.p", 'rb') as pfile: #load first-stage tree
    fTree = pickle.load(pfile)

#load norm const for database images
with open(rootfolder + "Dict_NormConst_01.p", 'rb') as pfile: #load first-stage tree
    alldict = pickle.load(pfile)

with open(rootfolder + "ScoreDict_01.p", 'rb') as pfile: #load first-stage tree
    scoredict = pickle.load(pfile)


#l = imgfiles[0] #first query image
l = rootfolder + "testing/mahatma_rice.jpg"
img = cv2.imread(l)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)

EntireImgWeights(fTree, descs, l)  #populate the query in the tree

constant = 0

N = 4536  #54*84

def QueryConst(node,l,N):
	global constant
	
	if(node.isLeaf==True or node.level == 1):				
		
		if((l in node.ifl) == True):
			N_i = len(node.ifl)
			constant = constant + (node.ifl[l] * np.log(N/N_i) )	# since all values are positive and Taking L-1 norm 	
		else:
			return						
		
		return
	else:
		for childs in node.children:
			QueryConst(childs,l,N)

QueryConst(fTree,l,N)  #norm const for "l" updated in the variable constant



def Scoring(node,l,N):	
	global scoredict
	global alldict
	global constant
	
	if(node.isLeaf==True or node.level == 1):
		
		if( (l in node.ifl) == False ):  #query not present
			return 		
		
		else:					
			for imgs in node.ifl:
				if(imgs == l):
					break
				else:
					c1 = alldict[imgs]  #norm const for db img
					c2 = constant  #norm const for query img
					N_i = len(node.ifl) 
					wi = np.log(N/N_i)
					
					qi = (node.ifl[l] * wi) / constant
					di = (node.ifl[imgs] * wi) / alldict[imgs]
					
					scoredict[imgs] = scoredict[imgs] + (abs(qi-di) - abs(qi) - abs(di))				
		return
				
	else:
		for childs in node.children:
			Scoring(childs,l,N)

Scoring(fTree,l,N)

with open(rootfolder + "ScoreDict_02.p", 'wb') as pfile1:
    pickle.dump(scoredict, pfile1)

x2 = sorted(scoredict, key=scoredict.__getitem__)


