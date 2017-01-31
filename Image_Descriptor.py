import cv2

#We are using BOWImgDescriptorExtractor Class to get the histogram of an image using a extracted Vocabulary.(Which was in turn created using BOWKMeansTrainer)

#Dextractor
SIFT = cv2.xfeatures2d.SIFT_create()

#DMatcher
bf = cv2.BFMatcher()

ImgDescEx = cv2.BOWImgDescriptorExtractor(SIFT,bf)

