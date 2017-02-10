#Function to calculate no. of Robust Matches, given 2 images, Query Image and Train Image(in DB)
#Input - Query Image Descriptors, Train Image Descriptors
import cv2

#assuming that stop words have been removed from the des1 and des2 descriptors set

def RatioTest(des1,des2,ratio=0.7):
	des1 = des1.astype(np.float32)
	des2 = des2.astype(np.float32)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)   #crossCheck=True stores only those matches which match both ways
	matches = bf.knnMatch(des1,des2,k=2)    			  #taking best 2 matches
	
	# store all the good matches as per Lowe's ratio test.
	good = []
	
	for m,n in matches:
		if m.distance < ratio * n.distance:
			good.append(m)

	return len(good)


def LoadDescriptors(pathlist):
	temp = []

	for t in pathlist:
		x1 = np.load(pathlist)
		temp.append(x1)

	return temp



def FilterDescAndKP(temp_A,Vocab,StopWords):
	bf = cv2.BFMatcher()

	return_a = []
	
	i = 0
	for t in temp_A:
		t = t.astype(np.float32)
		temp_matches = bf.match(t,Vocab)

		temp_del = [a.queryIdx for a in temp_matches if StopWords[a.trainIdx] == True]
		temp_del = temp_del.sort()     #sorting the list of descriptor entries to be deleted, so that they can be deleted easily

		temp_del = np.asarray(temp_del)
		
		count = 0

		while count < len(temp_del):
			np.delete(t, temp_del[count] - count , axis = 0)
			count+=1

		return_a.append(t)
	return return_a


def ReRank(pathlist, tets_desc):
	temp_A = LoadDescriptors(pathlist)   #Load DB Image descriptor into a list

	#Comment This block if Stop Word Removal is not to be performed before Lowe's Ratio Test
	Vocab = np.load(VocabPath) #Defined In GlobalVariables
	TotalImages = np.load(TotImagesPath) #Defined In GlobalVariables
	Vocab = Vocab.astype(np.float32)
	FreqList = np.load(FreqPath) #Defined In GlobalVariables
	StopWords = FreqList > FreqLimit* TotalImages
	temp_A = FilterDescAndKP(temp_A,Vocab,StopWords)   #Vocab is Vocabulary and Freq is Word Frequencies
	temp_test = [tets_desc]
	temp_test = FilterDescAndKP(temp_test,Vocab,StopWords)
	tets_desc = temp_test[0]

	ScoreList  = []

	i = 0
	while i < len(pathlist):
		ScoreList.append( [l,RatioTest(test_desc, temp_A[i])])

	ScoreList.sort(key = lambda:x x[1])

	return ScoreList