from tree_cluster import *

brchK = 10
depth_level = 3

RecursiveClustering(obs,brchK,depth_level)

lfnodes = np.asarray(leafnodes)
vocab_filename = "vocab_test01.npy"
np.save(vocab_filename,lfnodes)      # Save the raw(unweighted) vocabulary
