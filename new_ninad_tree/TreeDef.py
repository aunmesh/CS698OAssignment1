import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

class Tree:
	def __init__(self, value=None ,children=None):
		self.value=value  #centroids
		self.parent = []
		self.children = []	
		self.level = []
		self.isLeaf=True
		
		#self.Ncount = 0  #counting N_i variable		
		#self.Ncount_ForImage = False
		
		self.ifl = {} #empty dictionary
		
		if children is not None:
			for child in children:
				self.add_child(child)				
					
	def add_child(self,node):
		#assert isinstance(node, Tree)
		self.children.append(node)
		node.parent = self
		self.isLeaf = False
		
brch = 10
levels = 8    # Number of leaf nodes will be big-O: O(brch^(levels-1)) as first level is the trivial level
xvar = 0 # just a debugging variable

km = MiniBatchKMeans(n_clusters=brch, init='k-means++', n_init=10)     #Mini Batch #No batch size used.
#km = KMeans(n_clusters=brch, init='k-means++', n_init=10, n_jobs=-1)  

def TreeMake(obs,brch,levels):
	global km
			
	if(levels==0):
		return
	
	else:	
		p=Tree()
		p.level = levels
		
		print("At level:",levels)
		print(len(obs))
		
		if(len(obs)<= brch or levels==1):
			p.value = -1
			p.isLeaf = True						
			return p
			
		else:				
			model = km.fit(obs)
			centroids = model.cluster_centers_
			labids = model.labels_
			p.value = centroids			
			
			for i in range(brch):
				tmpobs =  obs[(labids == i)]		#calc according to cluster id
				tmptree = TreeMake(tmpobs,brch,levels-1) #Recursive Call
				p.add_child(tmptree)
				del tmpobs
	
			return p


print("\n Definitions loaded \n")
