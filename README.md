## Assignment 1 of CS698O(Visual Recognition)

-----------------

### Methods for improvement 

- Implement VLAD
- Higher order information about clusters and better normalization techniques
- Spatial verificaition (using pyramid match kernel?)
- Query expansion (but it can only be used if our top results are going to be good)

-----------------

### Instance Recognition

**Problem Statement:** 
Given a dataset of images of several objects your job is to train a model on it such that it is able to retrieve all the instances of a given test object. More specifially the given dataset consists of images of several objects and multiple instances of each object. When tested on an image/object your model's task is to retrieve all instances of the image in the dataset. In other words your model must give a similarity ranking on all images in the dataset such that instances of the object being tested have the highest rankings.

You can divide the given dataset into training and test sets in whichever manner you see fit. Actual testing will be done on an undisclosed dataset.

This assignment will be similar to a competition to see which group does the best. Grading will also be affected by this. We will also have a leader board at the end.


**Evaluation Metric:**
Mean Average Precision on the Top K images retrieved. [(Video explaining MAP)](https://www.youtube.com/watch?v=pM6DJ0ZZee0)


**Dataset:** [dataset](http://web.cse.iitk.ac.in/users/cs676/2017_visrec/www/asm1/Dataset.tar.gz)


**Submission format:** A zip file containing all of your code along with a readme file. Also include a report on your approach and any testing results you obtained.
Submission deadline: 10th February, 2017


**Clarifications:**

-	 60% of the grade will depend on the demo and 40% on the leaderboard performance
    
-    At the time of demo, you should bring your trained model , which should take a test image as input and generate a text file which will describe the ranking of the images for that test image.
    
-    The images of the category same as the test image should be retrieved before other categories. Amongst the images of same category, no particular order is required.
    
-    We will check MAP using top-k images in your ranking order. K can be variable so you should give ranking for all the images
    
-    The objects in the test set need not be centered, you can assume that the category of the test image will be present in the database.
    
-    The format of the text file is attached here. It should have file names and their categories on each newline.
    
-    If your code takes test image named "test1.jpg" as input, name the output file "test1.txt"
    
-    Submit the assignment on moodle. Only one persom should submit from each group. Make sure that the code you submit generates the output file in specified format.
    
-    Ask all the queries on the discussion forum on moodle
    
-    [Sample output file](http://web.cse.iitk.ac.in/users/cs676/2017_visrec/www/asm1/imagename.txt)

-----------------


