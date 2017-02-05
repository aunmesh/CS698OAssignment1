import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from TreeDef import * 

#obs = np.load("train_desc.npy")
obs = np.load("test_desc.npy")
obs = obs.astype(np.float32)		
		
finalTree = TreeMake(obs,brch,levels)

# Saving the constructed Tree
import pickle
with open('tree_pickle_01.p', 'wb') as pfile:  #first-stage
    pickle.dump(finalTree, pfile)
