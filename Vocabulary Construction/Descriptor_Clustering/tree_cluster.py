
#import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

obs = np.load("train_desc.npy")
obs = obs.astype(np.float32)


branch = 3
lev = 3
depth = 0 # just a debugging variable

#leafnodes = np.empty([brch**(levels-1),128])

global leafnodes

leafnodes = np.zeros([1,128])
leafnodes = list(leafnodes)

def RecursiveClustering(obs, brch, levels):
	global depth
	global leafnodes		
	
	km = MiniBatchKMeans(n_clusters=brch, init='k-means++', n_init=3, init_size=1000, batch_size=1000)
	#print("At level:",levels)
	depth = depth + 1
	
	if(levels>0):
		
		if(len(obs)<=brch):
			return
		else:
			mod01 = km.fit(obs) #perfrom clustering
		
			print("At level:",levels)
			print(len(obs))
			depth = depth + len(obs)
		
			if(levels == 1):
				cluscent = mod01.cluster_centers_ # need to STORE  the CLUSTER-CENTERS for EACH level in a TREE
			
				for i in range(len(cluscent)):
					leafnodes.append(list(cluscent[i]))
			
			#leafnodes = np.concatenate((leafnodes,cluscent),0)
			#leafnodes[count1:count2] = cluscent
					
		
			labids = mod01.labels_  # cluster-label for each point 
		
			for i in range(brch):			
				#tmpbool = (labids == i)
				tempobs =  obs[(labids == i)]		#calc according to cluster id
				#print(i)
				RecursiveClustering(tempobs, brch, levels-1)
				del tempobs
	
	else :
		return		


#lfnodes = np.asarray(leafnodes)
#np.save("vocab_500k.npy",lfnodes)      # Save the raw(unweighted) vocabulary







# Old code dumps

''' Dump1
def kmeancluster(obs, numclus = 3):
	km = MiniBatchKMeans(n_clusters=numclus, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
	mod10 = km.fit(obs) #perform the clustering	
	return mod10
'''



''' Dump2

sys.path.append(os.getcwd())
from globalvariables import * #script storing global variables such as location of root folder of image database
subdir = glob(rootfolder + "/*/") #rootfolder is specified in global variables without a trailing /



# Descriptor extractor class pointer...
# SIFT = cv2.xfeatures2d.SIFT_create
# dmatecher

####################### TRAINING CODE #############################

import cv2
import numpy as np

temp1 = np.load("test_desc.npy")
obs = temp1.astype(np.float32)
dictionarySize = 10000 # number of clusters
a = "./VisualWords_10k.npy" # final cluster centers

BOW = cv2.BOWKMeansTrainer(dictionarySize)
BOW.add(obs)

dictionary = BOW.cluster()
np.save(a, dictionary)

##################################################################


for p in training_paths:
    image = cv2.imread(p)
    gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, dsc= sift.detectAndCompute(gray, None)
    BOW.add(dsc)
for temp in subdir:
	os.chdir(temp)
	descfiles = glob(temp + "*.jpg*")
	
	for l in descfiles:
		print("loop starts") 
		img = cv2.imread(l)
		gray= cv2.cvtColor(img,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		#sift = cv2.xfeatures2d.SIFT_create()
		(kps, descs) = sift.detectAndCompute(gray, None)
		BOW.add(descs)
		
		#imgname = l.split(".")[-2]
		#print(imgname)
		#np.save(imgname,descs)

#dictionary created
dictionary = BOW.cluster()
				
a = rootfolder + "/DictSome.npy"	
np.save(a, dictionary)


'''


