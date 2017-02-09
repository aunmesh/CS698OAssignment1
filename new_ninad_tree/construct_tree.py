import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from TreeDef import * 
from globvar import *

obs = np.load(rootfolder + "train_desc.npy")
#obs = obs.astype(np.float32)		
		
#brch=10		
#levels=8 in TreeDef file
		
finalTree = TreeMake(obs,brch,levels)

#ALWAYS SAVE THE TREE IN A NEW FILE
import pickle
with open(rootfolder + "tree_10M_01.p", 'wb') as pfile:  #first-stage #ALWAYS SAVE THE TREE IN A NEW FILE
    pickle.dump(finalTree, pfile)

print("\n Clustering over \n")
