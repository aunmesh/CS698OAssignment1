README

There are two main folders for our code: 

1) The folder 'desc_codes' is used for extracting and concatenating the 
SIFT descriptors of the images used for training.

1.1) File 'desc_extract.py' goes to each object folder in the training 
set, sequentially loads every image, calculates the image's SIFT
descriptor and saves in the object folder with the same name as that of 
the image (with an extra '_desc_SIFT.npy' appended to it.

1.2) File 'concat_desc.py' is used for concatenating the descriptors to
form a combined dataset 'Desc_SIFT_all.npy' which will be then used for
the Vocabulary tree construction.


========================================================================


2) The folder 'tree_codes' has the tree class definitions and files that 
are used for hierarchical clustering of the combined descriptor dataset.

2.1)











