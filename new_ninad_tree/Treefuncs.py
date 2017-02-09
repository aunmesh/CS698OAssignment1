import numpy as np
import pickle
from TreeDef import *

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
	#leaflist = []
	for i in range(len(descmat)):
		FindWeights(node,descmat[i],imgname)
		#leaflist.append(FindWeights(node,descmat[i],imgname))
	return
	#return leaflist


# DEFINITIONS LOADED
print("\n Treefuncs loaded \n")


