import numpy as np

RootFolder = "/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/data_new/train_data"
VocabPath = "/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/data_new/vocab_100k.npy"
VocabFreqPath = "/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/data_new/BoW_Freq_Corrected.npy"
TotImagesPath = "/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/data_new/Tot_DB_Images.npy"

TestImgPath = '/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/CS698OAssignment1/TestImage_HistogramMatching/N3_327.jpg'
OutputPath = '/home/aunmesh/Academics/8th semester/CS698o-Visual Recognition/Assignment1/CS698OAssignment1/TestImage_HistogramMatching/'

Vocab = np.load(VocabPath)
VocabFreq = np.load(VocabFreqPath)
TotalImages = np.load(TotImagesPath)
