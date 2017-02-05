import numpy as np

import pickle
from TreeDef import *

import sys
import os
from glob import glob
from globvar import *

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
		
		'''
		if (childs[idx].Ncount_ForImage == False):
			childs[idx].Ncount = childs[idx].Ncount + 1 #update it.
			childs[idx].Ncount_ForImage = True  #Set it back to False after an Image is done with the tree.
		'''

def EntireImgWeights(node,descmat,imgname):  #Pass all the descriptors of an image 
	leaflist = []
	for i in range(len(descmat)):
		#FindWeights(node,descmat[i],imgname)
		leaflist.append(FindWeights(node,descmat[i],imgname))
	
	return leaflist

# Printing the dictionaries at the leaves
def GoToLeaf(node):
	
	if(node.isLeaf==True or node.level == 1):
		print(node.ifl)
		return
	else:
		for childs in node.children:
			GoToLeaf(childs)



# Load the Tree 
with open('tree_pickle_01.p', 'rb') as pfile: #load first-stage tree
    fTree = pickle.load(pfile)

sys.path.append(os.getcwd())
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /

for temp in subdir:
	os.chdir(temp)
	descfiles = glob(temp + "*.npy")
	print("\n Directory loop starts \n") 
	
	for l in descfiles:
		print(l)		
		temp_desc = np.load(l)
		imgname = l.split(".")[-2] + ".jpg"				
		EntireImgWeights(fTree,temp_desc,imgname)


print("\n All ends \n")


#Save the updated tree in a new file
with open('tree_pickle_03.p', 'wb') as pfile:
    pickle.dump(fTree, pfile)

