#!/usr/bin/python

ODELS GO WITH TRAIN SET, IF THERE IS NO TRAIN SET THEN MODELS GO WITH TEST SET

sugarcane-train-1.t7                    training set - 1836
sugarcane-test-1.t7                     testing set - 925

Results:
results1-1.txt: 10 iterations - Acc
results1-2.txt: 500 iterations

Models:
model1-1.t7
model1-2.t7

Image size: 32x32
Class: Good     - central crop, size 200 x 200
Class: Avg      - central crop, size 200 x 200
Class: Poor     - original
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200

------------------------------------------------------
sugarcane-train-2.t7                    training set - 1836
sugarcane-test-2.t7                     testing set - 925

Results:
results2-1.txt  Learning rate 0.001
results2-2.txt  Learning rate 0.005
results2-3.txt  Train - train
results2-4-1.txt        #conv layer 4
results2-4-2.txt        #conv layer 6
results2-5.txt  kernel size = 13
results2-6.txt  kernel size = 3
results2-7.txt  No pooling
results2-8.txt  Overlap pool
results2-9.txt  Max pooling
results2-10.txt #FC layers = 5
results2-10-1.txt #FC layers = 1
results2-10-2.txt #neurons at FC
results2-11.txt Dropout
results2-12.txt Drop out with lenet 2
results2-13.txt new lenet net:add(nn.SpatialConvolution(1, 20, 5, 5))
results2-14.txt 3 batches

Models:
model2-1.t7
model2-2.t7
model2-3.t7
model2-4-1.t7
model2-4-2.t7
model2-5.t7
model2-6.t7
model2-7.t7
model2-8.t7
model2-9.t7
model2-10.t7
model2-10-1.t7
model2-10-2.t7
model2-11.t7
model2-12.t7
model2-13.t7
model2-14.t7

Image size: 32x32
Class: Good     - original image
Class: Avg      - original image
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200

------------------------------------------------------
sugarcane-train-3.t7                    training set - 1936
sugarcane-test-3.t7                     testing set - 925

Results:
result3-1.txt
result3-2.txt
result3-3.txt
result3-4.txt i = 1000, learning rate = 0.0015
result3-5.txt Dropout = 0.05
result3-6.txt Conv kernel 3x3
result3-7.txt Max pool 3x3
result3-8.txt Overlap max pool (3,3,2,2)
result3-9.txt No pooling
result3-10.txt FC 2048
result3-11.txt #FC = 5
result3-12.txt Lenet 2
result3-13.txt Lenet 1

Models:
model3-1.t7
model3-2.t7
model3-3.t7
model3-4.t7
model3-5.t7
model3-6.t7
model3-7.t7
model3-8.t7
model3-9.t7
model3-10.t7
model3-11.t7
model3-12.t7
model3-13.t7

Image size: 32x32
Class: Good     - original image
Class: Avg      - central crop, size 200 x 200
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200

------------------------------------------------------
sugarcane-train-4.t7                    training set - 1518
sugarcane-test-4.t7                     testing set - 763

Results:
results4-1.txt
Models:
model4-1.t7

Image size: 32x32
Class: Good     - original image
Class: Avg      - original image
Class: Poor     - original image

------------------------------------------------------

sugarcane-train-5.t7                    training set
sugarcane-test-5.t7                     testing set

Results:
results5-1.txt

Models:
model5-1.t7

Image size: 32x32
Class: Good     - original
Class: Avg      - central crop, size 128 x 128
------------------------------------------------------
sugarcane-train-6.t7                    training set
sugarcane-test-6.t7                     testing set

Results:
result6-1.txt
result6-2.txt #conv layers = 5
result6-3.txt kernel size = 9x9

Models:
	model6-1.t7
	result6-2.txt
	result6-3.txt

	Image size: 128x128
	Class: Good     - original image
	Class: Avg      - central crop, size 128 x 128
	Class: Poor     - original image
			- central crop, size 128 x 128
			- bottom left crop, size 128 x 128
			- bottom right crop, size 128 x 128
------------------------------------------------------

sugarcane-train-7.t7                    training set - 1412 ••
sugarcane-test-7.t7                     testing set - 709

Results:
results7-1.txt

Models:
model7-1.t7

Image size: 32x32
Class: Good     - original
Class: Avg      - original
------------------------------------------------------
sugarcane-train-8.t7                    training set - 
sugarcane-test-8.t7                     testing set - 709

Results:
result8-1.txt

Models:
model8-1.t7
Image size: 32x32

Train:
Class: Good    	- original
		 		- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
Class: Avg      - original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
Test:
Class: Good     - original
Class: Avg      - original

------------------------------------------------------

sugarcane-train-9.t7                    training set
sugarcane-test-9.t7                     testing set

Results:
result9-1.txt

Models:
model9-1.t7
Image size: 32x32

Training
Class: Good     - original image
Class: Avg      - central crop, size 200 x 200
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
Testing
Class: Good     - original image
Class: Avg      - original image 35%, central crop, size 200 x 200 65%
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
------------------------------------------------------

sugarcane-train-10.t7                    training set
sugarcane-test-10.t7                     testing set

Results:
result10-1.txt  lenet  1 i = 20
result10-2.txt  lenet  1 i = 1000

Models:
model10-1.t7
model3-5.t7
Image size: 32 x 32


Training
Class: Good     - original image
Class: Avg      - central crop, size 200 x 200
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
Testing
Class: Good     - original image
Class: Avg      - original image 35%, central crop, size 200 x 200 65%
Class: Poor     - original image
		- central crop, size 200 x 200
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
------------------------------------------------------

sugarcane-train-11.t7                    training set
sugarcane-test-11.t7                     testing set

Results:
result11-1.txt  lenet  1

Models:
model11-1.t7

Size: 128 x 128
Training
Class: Good     - original image
Class: Avg      - original image
Class: Poor     - original image
		- central crop, size 128 x 128
		- bottom left crop, size 128 x 128
		- bottom right crop, size 128 x 128
------------------------------------------------------
sugarcane-train-12.t7                    training set
sugarcane-test-12.t7                     testing set

Results:
result12-1.txt  lenet  1

Models:
model12-1.t7
Training
Size: 32 x 32
Class: Good     - original
		 - bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg      - original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Test:
Class: Good     - original
Class: Avg      - original
------------------------------------------------------
sugarcane-test-13.t7                     testing set

Size: 32 x 32
Class: Good     - original
		 - bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg      - original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
------------------------------------------------------
sugarcane-test-14.t7                     testing set

Size: 32 x 32
Class: Good
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200

------------------------------------------------------
sugarcane-train-15.t7                    training set
sugarcane-test-15.t7                    testing set

Size: 32 x 32
Class: Good		- original
		 		- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
Class: Avg		- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200

Class: Med		- original
				- bottom left crop, size 200 x 200
				- bottom right crop, size 200 x 200
				- central crop, size 200 x 200
------------------------------------------------------
sugarcane-test-16.t7                     testing set

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
------------------------------------------------------
sugarcane-train-17.t7                    train set
sugarcane-test-17.t7                    testing set

Size: 128 x 128
Class: Good     - original
		 - bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg      - original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200

------------------------------------------------------
sugarcane-test-18.t7                    testing set
Size: 128 x 128
Class: Good     - original
Class: Avg      - original

------------------------------------------------------
sugarcane-test-19.t7                    testing set

Size: 128 x 128
Class: Good
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
------------------------------------------------------
sugarcane-train-20.t7                    train set
sugarcane-test-20.t7                    testing set

Size: 128 x 128
Class: Good     - original
		 - bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg      - original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Poor     - original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
------------------------------------------------------
sugarcane-test-21.t7                    testing set
Size: 128 x 128
model 21-1
Class: Good     - original
Class: Avg      - original
Class: Poor      - original
------------------------------------------------------
sugarcane-test-22.t7                    testing set

Size: 128 x 128
model 22-1
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
------------------------------------------------------
sugarcane-test-23.t7                    training set

Size: 32 x 32
model 23 - 1
Class: Good
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
------------------------------------------------------
sugarcane-train-24.t7                    training set

Size: 32 x 32
model 24 - 1
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
------------------------------------------------------
sugarcane-train-25.t7                   training set
sugarcane-test-25.t7                    testing set

Size: 32 x 32
Growing season: Med
model 25 - 1

Class: Good     - original
Class: Avg      - original
------------------------------------------------------
sugarcane-train-26.t7                   training set
sugarcane-test-26.t7                    testing set

Size: 32 x 3
Growing season: Med
model 26 - 1

Train
Class: Good     - original
				- center crop, size 200 x 200
Class: Avg      - original
		- center crop, size 200 x 200
Test

Class: Good
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200

------------------------------------------------------
sugarcane-test-27.t7                    testing set
Size: 32 x 3
Growing season: Med
Class: Good     - Original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200
Class: Avg      - Original
		- bottom left crop, size 200 x 200
		- bottom right crop, size 200 x 200
		- central crop, size 200 x 200



