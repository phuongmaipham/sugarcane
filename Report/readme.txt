PART A: THE DATASET  

=========================================================================================================
============================================Raw data set=================================================
=========================================================================================================
The real dataset consists of 2281 ground level images of types png and jpg. 
They are segregated by the sugarcane age into three growing seasons: Early, Mid, and Late. 
Each category contains sample sugarcane images of different qualities (classes): Poor, Average and Good. 
The directory to each category is as follow:

early_good="/home/superuser/project/Picture/Early/Good"
early_avg="/home/superuser/project/Picture/Early/Average"
early_poor="/home/superuser/project/Picture/Early/Poor"
mid_good = "/home/superuser/project/Picture/Mid/Good"
mid_avg = "/home/superuser/project/Picture/Mid/Average"
mid_poor="/home/superuser/project/Picture/Mid/Poor"
late_good = "/home/superuser/project/Picture/Late/Good"
late_avg="/home/superuser/project/Picture/Late/Average"
late_poor="/home/superuser/project/Picture/Late/Poor"

=========================================================================================================
========================================Preprocessed dataset=============================================
=========================================================================================================
The preprocessed dataset consists of 1809 ground level images of types png and jpg. 
They are segregated by the sugarcane age into three growing seasons: Early, Mid, and Late. 
Each category contains sample sugarcane images of different qualities (classes): Poor, Average and Good. 
The directory to each category is as follow:

early_good="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Good"
early_avg="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Average"
early_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Poor"
mid_good = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Good"
mid_avg = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Average"
mid_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Poor"
late_good = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Good"
late_avg="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Average"
late_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Poor"

PART B: SOURCE CODE 

=========================================================================================================
==========================================read_images.lua================================================
=========================================================================================================
- Read all images and their labels:
	+ If “Good” is found in a file name, the image associated with it will be in class Good. 
	We will label it as “1”
	+ If “Average” is found in a file name, the image associated with it will be in class Average. 
	We will label it as “2”  
	+ If “Poor” is found in a file name, the image associated with it will be in class Poor. 
	We will label them as “3”
- Store 2/3 images in each directory in training set, the rest will be in testing set
- Training set and testing set are saved in t7 format (default format in Torch)

=========================================================================================================
==========================================train_model.lua================================================
=========================================================================================================
- Import the training and the testing set
- Train a model with the training set
- Test the trained model with the testing set
- Display the calculated possibility per class for each image
- Display testing accuracy and the percentage of true positive per class
- Save the trained model

=========================================================================================================
==============================================test.lua===================================================
=========================================================================================================
- Import a testing set and a model
- Apply the model to the testing set
- Display the calculated possibility per class for each image
- Display testing accuracy and the percentage of true positive per class

PART C: THE DATASETS IN T7 FORMAT AND THE TRAINED MODELS

=========================================================================================================
=========================================DATASET 1: ORIGNAL==============================================
=========================================================================================================
-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-4.t7                    training set - 1518
Models:
	model4-1.t7
sugarcane-test-4.t7                     testing set - 763
sugarcane-test-4b.t7                    testing set - 90, #Good = # Med = #Poor = 30 

Image size: 32x32
Class: Good     		- original image
Class: Avg      		- original image
Class: Poor    			- original image

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-15.t7                   training set - 6072
Models: 
	model 15 - 1
sugarcane-test-15.t7                    testing set - 3052

Size: 32 x 32
Class: Good			- original
		 		- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Med			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-24.t7                    training set - 4554 
Models: 
	model 24 - 1
sugarcane-test-16.t7                     testing set - 2289

Size: 32 x 32
Class: Good
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Poor
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-1.t7                    training set - 1836
Models:
	model1-1.t7
sugarcane-test-1.t7                     testing set - 925

Image size: 32x32
Class: Good			- central crop, size 200 x 200
Class: Avg			- central crop, size 200 x 200
Class: Poor			- original
				- central crop, size 200 x 200
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				
-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-2.t7                    training set - 1836
Models:
	model2-1.t7			Learning rate 0.001
	model2-2.t7			Learning rate 0.005
	model2-14.t7			3 batches
sugarcane-test-2.t7                     testing set - 925

Image size: 32x32
Class: Good			- original image
Class: Avg			- original image
Class: Poor			- original image
				- central crop, size 200 x 200
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
-----------------------------------------------CLASS ALL-------------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-3.t7                    training set - 1936
Models:
	 model3 - 1.t7 
sugarcane-test-3.t7                     testing set - 925

Image size: 32x32
Class: Good			- original image
Class: Avg			- central crop, size 200 x 200
Class: Poor			- original image
				- central crop, size 200 x 200
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				

*********************************************************************************************************
-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-7.t7                     training set - 1412 
Models:
	model7-1.t7
sugarcane-train-7b.t7  			 training set - #Good = #Med = 653; #Total = 1306 
Models: 
	model7b - 1.t7
sugarcane-test-7.t7                      testing set - 709
sugarcane-test-7b.t7                     testing set - #Good = #Med = 122; #Total = 366 

Image size: 32x32
Class: Good     - original
Class: Avg      - original

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-8.t7                     training set - 2824
Models:
	model8-1.t7

Image size: 32x32
Class: Good    			- original
		 		- center crop, size 200 x 200
Class: Avg			- original
				- center crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-12.t7                    training set - 5648
Models:
	model12-1.t7			 lenet	1
	model12-2.t7			 lenet	2
sugarcane-train-12b.t7                   training set - #Good = #Avg = 2608; #Total = 5216
Models:
	model12b-1.t7			 lenet 1 
sugarcane-test-13.t7                     testing set - 2836

Size: 32 x 32
Class: Good			- original
		 		- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-23.t7                    training set - 4236 
Models:
	model23 - 1.t7 
sugarcane-test-14.t7                     testing set - 2127
	
Size: 32 x 32
Class: Good
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-18.t7                    testing set - 709 
Size: 128 x 128
Class: Good     - original
Class: Avg      - original

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-17.t7                   training set - 5648
Model 
	model17-1.t7 i = 15 
	model17-2.t7 i = 30
sugarcane-test-17.t7                    testing set - 2386 

Size: 128 x 128
Class: Good			- original
		 		- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-19.t7                    testing set - 2127 

Size: 128 x 128
Class: Good
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

*********************************************************************************************************
*********************************************************************************************************
-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-25.t7                   training set - 666
Model:
	model25 - 1.t7 
sugarcane-train-25b.t7                  training set - #Good  = #Avg = 282; #Total = 564
Model: 
	model25b - 1.t7 
sugarcane-test-25.t7                    testing set - 333

Size: 32 x 32
Class: Good     		- original
Class: Avg      		- original

-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-26.t7                   training set - 1332
Model:
	model26 - 1.t7 

Size: 32 x 32 
Class: Good     		- original
				- center crop, size 200 x 200
Class: Avg      		- original
				- center crop, size 200 x 200

-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-28.t7                   training set - 2664 
Model:
	model28 - 1.t7 			Lenet 1
	model28 - 2.t7 			Lenet 2 
sugarcane-train-28b.t7                  training set -  #Good = #Avg = 564; #Total = 1128
Model: 
	model28b - 1.t7 
sugarcane-test-27.t7                    testing set - 1332

Size: 32 x 32
Class: Good			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
				
-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-26.t7                    testing set - 999

Size: 32 x 32 
Class: Good
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200	

-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-36.t7                    testing set - 333

Size: 128 x 128
Class: Good     		- original
Class: Avg      		- original
					
-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-35.t7                   training set - 2664
Model:
	 model35 - 1.t7 		Lenet 1
sugarcane-test-35.t7                   	testing set - 1332  

Size: 128 x 128 
Class: Good			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

-----------------------------------------GROWING SEASON: MED---------------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-37.t7                   	 testing set - 999   

Size: 128 x 128 
Class: Good		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

*********************************************************************************************************
*********************************************************************************************************
-------------------------------------GROWING SEASON: LATE & EARLY----------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-29.t7                   training set - 746
Model:
	model29 - 1.t7 
sugarcane-train-29b.t7                  training set - #Good = #Avg = 370; #Total = 740
Model:
	model29b - 1.t7 
sugarcane-test-29.t7                    testing set - 376 

Size: 32 x 32
Class: Good     		- original
Class: Avg      		- original

-------------------------------------GROWING SEASON: LATE & EARLY----------------------------------------
----------------------------------------CLASS "GOOD" & "MEDIUM"------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-30.t7                   training set - 2984
Model:
	 model30 - 1.t7 
sugarcane-train-30b.t7                  training set - #Good = #Avg = 1480; #Total = 2960 
Model:
	model30b - 1.t7 
sugarcane-test-30.t7 			testing set - 1504 

Size: 32 x 32 
Class: Good			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

------------------------------------GROWING SEASON: LATE & EARLY-----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-31.t7 			testing set - 1128 

Size: 32 x 32 
Class: Good		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

------------------------------------GROWING SEASON: LATE & EARLY-----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-34.t7                    testing set - 376 

Size: 128 x 128
Class: Good     		- original
Class: Avg      		- original

------------------------------------GROWING SEASON: LATE & EARLY-----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-32.t7                   training set - 2984 
Model:
	model30 - 1.t7 
sugarcane-test-32.t7 			testing set - 1504 

Size: 128 x 128
Class: Good			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg			- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
				
------------------------------------GROWING SEASON: LATE & EARLY-----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-33.t7 			testing set - 1128 

Size: 128 x 128
Class: Good		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg		
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
			
=========================================================================================================
======================================DATASET 1: PREPROCESSED============================================
=========================================================================================================
-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-38.t7			training set - 1203 
Model:
	model38-1.t7			lenet1, i = 15 
	model38-2.t7			lenet1,i = 50 
sugarcane-test-38.t7                    testing set - 605

Size: 32 x 32
Class: Good			- original
Class: Avg			- original

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-36.t7                   training set - 4812 
Models:
	model36-1.t7			lenet 1
	model36-2.txt  			lenet	1, #conv layer = 4
sugarcane-test-52.t7                    testing set - 2420

Size: 32 x 32
Class: Good			- original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- original
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-39.t7                   training set - 3609 
Model:
	model39-1.t7			lenet	1
	model39-2.t7			lenet	1	#conv layer = 4 
sugarcane-test-39.t7                    testing set - 1815 

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
sugarcane-train-52.t7                   training set - 1203
Model:
	model52-1.t7			lenet 1, i = 15
	model52-2.t7			lenet 1, i = 50
sugarcane-test-53.t7			testing set - 605
	
Size: 32 x 32
Class: Good			- squared original
Class: Avg			- squared original

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
sugarcane-train-53.t7                   training set - 4812
Model:
	model53-1.t7			lenet1, i = 15
	model53-2.t7			lenet1, i = 50

Size: 32 x 32
Class: Good			- squared original 
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- squared original 
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
		
*********************************************************************************************************
-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-48.t7                    testing set - 605 

Size: 32 x 32
Class: Good			- Original
Class: Avg			- Original

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-48.t7                   training set - 4812 
Models:
	model48-1.t7			lenet	1 i = 15 

Size: 32 x 32
Class: Good			- Original
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg			- Original
				- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448

-----------------------------------------GROWING SEASON: ALL---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-49.t7                   training set - 3609 
Model:
	model49-1.t7
sugarcane-test-49.t7                    testing set - 1815 

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg
				- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448

*********************************************************************************************************
*********************************************************************************************************
-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-55.t7 			training set - 540 
Model:
	model55-1.t7 			lenet1, i = 15
	model55-2.t7			lenet1, i = 50 
sugarcane-test-41.t7                    testing set - 271 

Size: 32 x 32
Class: Good			- Original
Class: Avg			- Original

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-41.t7                   training set - 2160
Model:
	model41-1.t7  			lenet	1 i = 15 
	model41-2.t7  			i = 30 
	model41-3.t7  			lenet	1 # conv layers = 4 
sugarcane-test-42.t7                    testing set - 1084

Size: 32 x 32
Class: Good			- Original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- Original
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-40.t7                   training set - 1620 
Model:
	model40-1.t7  			lenet	1
	model40-2.t7  			lenet	1		#conv layers = 4
sugarcane-test-40.t7                    testing set - 813

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

*********************************************************************************************************
-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-46.t7                    testing set - 271 

Size: 32 x 32
Class: Good			- Original
Class: Avg			- Original

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-54.t7                    training set - 540 
Models:
	model54-1.t7			 lenet 1, i = 50
	model54-2.t7			 lenet 1, i = 100
	model54-3.t7			 lenet 1, i = 15
	model54-4.t7			 lenet 1, i = 30
sugarcane-test-54.t7                     testing set - 271

Size: 32 x 32
Class: Good			- Squared original
Class: Avg			- Squared original

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-56.t7                    training set - 2160 
Models:
	model56-1.t7			 lenet 1, i = 15
	model56-2.t7			 lenet 1, i = 50

Size: 32 x 32
Class: Good			- Squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- Squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-46.t7                   training set - 2160
Models:
	model46-1.t7			lenet	1 i = 15 

Size: 32 x 32
Class: Good			- Original
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg			- Original
				- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448

-----------------------------------------GROWING SEASON: MED---------------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-47.t7                   training set - 1620 
Model:
	model47-1.t7			lenet	1
sugarcane-test-47.t7                    testing set - 813

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg
				- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448

*********************************************************************************************************
*********************************************************************************************************
-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-59.t7			training set - 663
Models:
	model59-1.t7 			lenet1, i = 15
	model59-2.t7 			lenet1, i = 50 
sugarcane-test-43.t7                    testing set -  334

Size: 32 x 32
Class: Good			- Original
Class: Avg			- Original

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-43.t7                   training set - 2652
Model:
	model43-1.t7  			lenet	1
	model43-2.t7  			lenet	1	# conv layers = 4 
sugarcane-test-45.t7                    testing set -  1336 

Size: 32 x 32
Class: Good			- Original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- Original
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-44.t7                   training set - 1989 
Model:
	model44-1.t7  			lenet	1
	model44-2.t7  			lenet	1 # conv layers = 4
sugarcane-test-44.t7                    testing set -  1002 

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg		
				- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-test-50.t7                    testing set -  334

Size: 32 x 32
Class: Good			- Original
Class: Avg			- Original

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-57.t7                   testing set -  663
Models:
	model57-1.t7 			lenet1, i = 15
	model57-2.t7 			lenet1, i = 50
sugarcane-test-57.t7                    testing set -  334

Size: 32 x 32
Class: Good			- Squared original
Class: Avg			- Squared original

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-58.t7                   testing set - 2652
Models:
	model58-1.t7 			lenet1, i = 15
	model58-2.t7 			lenet1, i = 50

Size: 32 x 32
Class: Good			- Squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- Squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-50.t7                   training set - 2652
Model:
	model50-1.t7			lenet	1
	
Size: 32 x 32
Class: Good			- Original
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg			- Original
				- bottom left crop, size 448x448
				- bottom right crop, size 4448x448
				- central crop, size 448x448

-------------------------------------GROWING SEASON: EARLY & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-51.t7                   training set - 1989 
Model:
	model51-1.t7  			lenet	1
sugarcane-test-51.t7                    testing set -  1002 

Size: 32 x 32
Class: Good	
		 		- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448
Class: Avg		
				- bottom left crop, size 448x448
				- bottom right crop, size 448x448
				- central crop, size 448x448

-------------------------------------GROWING SEASON: MID & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-60.t7                   testing set -  765
Models:
	model60-1.t7 			lenet1, i = 15
	model60-2.t7 			lenet1, i = 50
sugarcane-test-60.t7                    testing set -  385

Size: 32 x 32
Class: Good			- original
Class: Avg			- original

-------------------------------------GROWING SEASON: MID & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-64.t7                   testing set - 2295 
Models:
	model64-1.t7 			lenet1, i = 15
sugarcane-test-64.t7                    testing set - 1155

Size: 32 x 32
Class: Good			
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg		
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-------------------------------------GROWING SEASON: MID & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-63.t7                   testing set -  3060
Models:
	model63-1.t7 			lenet1, i = 15
sugarcane-test-63.t7                    testing set -  1540

Size: 32 x 32
Class: Good			- original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

-------------------------------------GROWING SEASON: MID & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-61.t7                   testing set -  765
Models:
	model61-1.t7 			lenet1, i = 15
	model61-2.t7 			lenet1, i = 50
sugarcane-test-61.t7                    testing set -  385

Size: 32 x 32
Class: Good			- squared original
Class: Avg			- squared original


-------------------------------------GROWING SEASON: MID & LATE----------------------------------------
---------------------------------------CLASS "GOOD" & "MEDIUM"-------------------------------------------
---------------------------------------------------------------------------------------------------------
sugarcane-train-62.t7                   testing set -  3060
Models:
	model62-1.t7 			lenet1, i = 15
	model62-2.t7 			lenet1, i = 50

Size: 32 x 32
Class: Good			- squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400
Class: Avg			- squared original
		 		- bottom left crop, size 400 x 400
				- bottom right crop, size 400 x 400
				- central crop, size 400 x 400

