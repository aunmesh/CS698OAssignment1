#Function to calculate no. of Robust Matches, given 2 images, Query Image and Train Image(in DB)

#Input - Query Image Descriptors, Train Image Descriptors
import cv2

#assuming that stop words have been removed from the des1 and des2 descriptors set
def RatioTest(des1,des2,ratio=0.7):
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)   #crossCheck=True stores only those matches which match both ways
	matches = bf.knnMatch(des1,des2,k=2)    			  #taking best 2 matches
	

	# store all the good matches as per Lowe's ratio test.
	good = []
	
	for m,n in matches:
	    if m.distance < ratio * n.distance:
	        good.append(m)

	return good












