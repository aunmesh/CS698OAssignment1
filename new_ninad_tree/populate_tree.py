import numpy as np

import pickle
from TreeDef import *

import sys
import os
from glob import glob
from globvar import *

from Treefuncs import *


# Load / OPEN the Tree ##CHECK TREE NAME
with open(rootfolder + "tree_1M_01.p", 'rb') as pfile: #load first-stage tree
    fTree = pickle.load(pfile)

sys.path.append(os.getcwd())
subdir = glob(trainfolder + "/*/") #rootfolder is specified in global variables without a trailing /
# CHECK DEFINITION OF ROOT FOLDER AND ADJUST ACCORDINGLY

alldict={}
scoredict={}

for temp in subdir:
	os.chdir(temp)
	descfiles = glob(temp + "*.npy")
	print("\n Directory loop starts \n") 
	
	for l in descfiles:
		#print(l)		
		temp_desc = np.load(l)
		imgname = l.split(".")[-2] + ".jpg"	
		alldict[imgname]=0 #initialised with zero
		scoredict[imgname]=0 #initialised with zero			
		EntireImgWeights(fTree,temp_desc,imgname)

print("\n All ends \n")

#SAVE the updated tree in a NEW file
with open(rootfolder + "tree_1M_02.p", 'wb') as pfile:
    pickle.dump(fTree, pfile)

with open(rootfolder + "Dict_NormConst_01.p", 'wb') as pfile2:
    pickle.dump(alldict, pfile)

with open(rootfolder + "ScoreDict_01.p", 'wb') as pfile1:
    pickle.dump(scoredict, pfile1)






'''
def FindWeights(node,dvect,imgname):
	
	if(node.isLeaf==True or node.level == 1):
				
		if((imgname in node.ifl) == False):  #Check if an instance exists
			tempdict = {imgname:1} #set initial to one
			node.ifl.update(tempdict)		
			del tempdict # OR tempdict.clear()
		else:			
			node.ifl[imgname] = node.ifl[imgname] + 1	#updating n_i		
		
		return node
		
	else:
		centroids = node.value
		idx = np.linalg.norm((centroids-dvect), axis=1).argmin()		
	
		childs = node.children
		FindWeights(childs[idx], dvect, imgname)
		

def EntireImgWeights(node,descmat,imgname):  #Pass all the descriptors of an image 
	leaflist = []
	for i in range(len(descmat)):
		#FindWeights(node,descmat[i],imgname)
		leaflist.append(FindWeights(node,descmat[i],imgname))
	
	return leaflist


# Printing the dictionaries at the leaves
def GoToLeaf(node):
	#global countlist
	if(node.isLeaf==True or node.level == 1):
		print(node.ifl)		
		print("\n")
		#countlist.append(1)
		return
	else:
		for childs in node.children:
			GoToLeaf(childs)


## DEFINITIONS LOADED
'''

