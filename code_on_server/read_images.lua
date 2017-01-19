require 'nn';
require 'image';
require 'torchx';
cutorch = require 'cutorch'
local data_aug = require("data_aug")
--require('trepl')()
--DATASET
early_good="/home/superuser/project/Picture/Early/Good"
early_avg="/home/superuser/project/Picture/Early/Average"
early_poor="/home/superuser/project/Picture/Early/Poor"
mid_good = "/home/superuser/project/Picture/Mid/Good"
mid_avg = "/home/superuser/project/Picture/Mid/Average"
mid_poor="/home/superuser/project/Picture/Mid/Poor"
late_good = "/home/superuser/project/Picture/Late/Good"
late_avg="/home/superuser/project/Picture/Late/Average"
late_poor="/home/superuser/project/Picture/Late/Poor"

-- bl, br, c, bright+origin, contrast+origin, 

--add filenames in a folder to a table for readability  
function dir_to_list(dir)
	label = {}
	files = {}
	--create a list of images in a directory 
	for file in paths.files(dir) do
		if (file:find('jpg') or file:find('JPG') or file:find('png') or file:find('PNG') or file:find('jpeg') or file:find('JPEG')) then
		table.insert(files,paths.concat(dir,file))
		end
	end
--	print (files)
--	print (#files)
	return files
end

ivch = 3                --#channel
--desImaX = 45            --w
--desImaY = 45            --h
desImaX = 32
desImaY = 32

trainData = {
                data = torch.Tensor(2054, ivch,desImaX,desImaY),		-- old vvalue: 1518 (x1), 3081 (x2)
                labels = torch.Tensor(2054),
--                size = function() return #files end 
            }
testData =  {
                data = torch.Tensor(1027, ivch,desImaX,desImaY),		-- old value : 763 (x1), 1526 (x2)
                labels = torch.Tensor(1027),
--                size = function() return #files end
            }

--convert images to .t7 file, create train.t7 and test.t7
function image_to_t7(my_files,index_train,index_test)
	files = {}
	files = dir_to_list(my_files)
	images = {}
	--cutorch.setDevice(1)
	--label1 = {}

	--create a training dataset
	--label = 1 if Good, 2 if Avg and 3 if Poor
	train_size, test_size = split_dataset(files)
	print (train_size)	
	--training data
	for i,file in ipairs(files) do
		print ('i',i)
		print ('file', file)
		if(i<=train_size) then	
			print ('index_train',index_train)
			my_image = image.load(file)
			clone_image = my_image
			my_image = image.scale(my_image,32,32,'bilinear')
			--my_image:resize(3,45,45)
			trainData.data[index_train] = my_image
			--table.insert(images, my_image)
			if file:find('Good') then
				--label1[i] = 1
				trainData.labels[index_train] = 1
			else if file:find('Average') then
				--label1[i] = 2
				trainData.labels[index_train] = 2
			else
				--label1[i] = 3
				trainData.labels[index_train] = 3
			end
			end
			print (trainData.labels[index_train])
			true_label = trainData.labels[index_train]		-- true label for this batch of clones
			index_train = index_train + 1
            -- increase #input images (center 500, resize 200 )
			-- CENTER CLONE
			--[[
            clone_c = image.crop(clone_image,"c",200, 200)
			clone_c = image.scale(clone_c,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			--]]
			
		else if(i>train_size)then
            print ('index_test',index_test)
            my_image = image.load(file)
			clone_image = my_image
			my_image = image.scale(my_image,32,32,'bilinear')
                        --my_image:resize(3,45,45)
                        testData.data[index_test] = my_image
                        --table.insert(images, my_image)
                        if file:find('Good') then
                                --label1[i] = 1
                                testData.labels[index_test] = 1
                        else if file:find('Average') then
                                --label1[i] = 2
                                testData.labels[index_test] = 2
                        else
                                --label1[i] = 3
                                testData.labels[index_test] = 3
                        end
                        end
			true_label = testData.labels[index_test]              -- true label for this batch of clones
            index_test = index_test + 1		
            -- increase #input images (center 500, resize 200 )
            -- CENTER CLONE
			--[[
            clone_c = image.crop(clone_image,"c",200, 200)
            clone_c = image.scale(clone_c,32,32,'bilinear')
            testData.data[index_test] = clone_c
            testData.labels[index_test] = true_label
            index_test = index_test + 1
			--]]
		end
		end		
	end
	--label = torch.Tensor(label1)
	--data = torch.Tensor(#images,3,45,45)
	--print (#images)
	--print (images)
	
	--for i=1,#images do
	--	data[i] = images[i]
	--	label[i] = label1[i]
	--end
	--Data_to_Write = { data = data, label = label }
--	image.display (trainData.data[1])
end 

-- since the size of poor class is relatively small compare to the other two, we will increase the size of poor class
function image_to_t7_poor (my_files,index_train,index_test) 
	files = {}
	files = dir_to_list(my_files)
	images = {}
	--create a training dataset
	--label = 1 if Good, 2 if Avg and 3 if Poor
	train_size, test_size = split_dataset(files)
	print (train_size)
	for i,file in ipairs(files) do
		print ('i',i)
		print ('file', file)
		--training data
		if(i<=train_size) then
			print ('index_train',index_train)
			my_image = image.load(file)
			clone_image_1 = image.load(file)
			clone_image_2 = image.load(file)
			clone_image_3 = image.load(file)
			my_image = image.scale(my_image,32,32,'bilinear')
			--my_image:resize(3,45,45)
			trainData.data[index_train] = my_image
			--table.insert(images, my_image)
			-- label = 'poor'
			trainData.labels[index_train] = 3
			print (trainData.labels[index_train])
			true_label = trainData.labels[index_train]              -- true label for this batch of clones
			index_train = index_train + 1
			-- increase #input images (center 500, resize 200 )
			-- CENTER CLONE
			clone_c = image.crop(clone_image_1,"c",200, 200)
			clone_c = image.scale(clone_c,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			-- BOTTOM LEFT CLONE
			clone_bl = image.crop(clone_image_1,"bl",200, 200)
			clone_bl = image.scale(clone_bl,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			-- BOTTOM RIGHT CLONE
			clone_br = image.crop(clone_image_1,"br",200, 200)
			clone_br = image.scale(clone_br,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			-- BRIGHTNESS AND CONTRAST ADJUSTMENT 1
			clone_bc1 = brightness(clone_image_2)
			clone_bc1 = image.scale(clone_bc1,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1		
			-- BRIGHTNESS AND CONTRAST ADJUSTMENT 2
			clone_bc2 = brightness(clone_image_3)
			clone_bc2 = image.scale(clone_bc2,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1						

			-- testing set 
		else if(i>train_size)then
			print ('index_test',index_test)
			my_image = image.load(file)
			clone_image_1 = image.load(file)
			clone_image_4 = image.load(file)
			clone_image_5 = image.load(file)
			my_image = image.scale(my_image,32,32,'bilinear')
			--my_image:resize(3,45,45)
			testData.data[index_test] = my_image
			--table.insert(images, my_image)
			-- label = 'poor'
			testData.labels[index_test] = 3
			true_label = testData.labels[index_test]              -- true label for this batch of clones
			index_test = index_test + 1
			-- increase #input images (center 500, resize 200 )
			-- CENTER CLONE
			clone_c = image.crop(clone_image,"c",200, 200)
			clone_c = image.scale(clone_c,32,32,'bilinear')
			testData.data[index_test] = clone_c
			testData.labels[index_test] = true_label
			index_test = index_test + 1
			-- BOTTOM LEFT CLONE
			clone_bl = image.crop(clone_image_1,"bl",200, 200)
			clone_bl = image.scale(clone_bl,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			-- BOTTOM RIGHT CLONE
			clone_br = image.crop(clone_image_1,"br",200, 200)
			clone_br = image.scale(clone_br,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1
			-- BRIGHTNESS AND CONTRAST ADJUSTMENT 1
			clone_bc1 = brightness(clone_image_4)
			clone_bc1 = image.scale(clone_bc1,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1		
			-- BRIGHTNESS AND CONTRAST ADJUSTMENT 2
			clone_bc2 = brightness(clone_image_5)
			clone_bc2 = image.scale(clone_bc2,32,32,'bilinear')
			trainData.data[index_train] = clone_c
			trainData.labels[index_train] = true_label
			index_train = index_train + 1					
		end
		end
	end     
end
-- return size of the training set and size of the testing set at each directory
function split_dataset (files) 
	--2/3 train, 1/3 test
	total_images = #files
	train_size = math.modf((total_images/3)*2)
	test_size = total_images - train_size
	return  train_size, test_size
end	

-- return new index of the training set and new index of the testing set after saving the images from the previous directory
function get_index (old_index_tr,old_index_te,train_size,test_size)
	new_index_tr = old_index_tr + train_size*2
	new_index_te = old_index_te + test_size*2
	return new_index_tr, new_index_te
end

--EARLY GOOD
files_eg = {}
files_eg = dir_to_list(early_good)		
--start with index = 1
image_to_t7(early_good,1,1)
train_size_eg, test_size_eg = split_dataset(files_eg)
print ('early good')
print (train_size_eg)
print (test_size_eg)

--EARLY AVG
files_ea = {}
files_ea = dir_to_list(early_avg)
index_train_ea,index_test_ea = get_index(1,1,train_size_eg,test_size_eg)
image_to_t7(early_avg,index_train_ea,index_test_ea)
train_size_ea, test_size_ea = split_dataset(files_ea)
print ('early aavg')
print (train_size_ea)
print (test_size_ea)

--EARLY POOR
files_ep = {}
files_ep = dir_to_list(early_poor)
index_train_ep,index_test_ep = get_index(index_train_ea,index_test_ea,train_size_ea, test_size_ea)
image_to_t7_poor(early_poor,index_train_ep,index_test_ep)
train_size_ep, test_size_ep = split_dataset(files_ep)
print ('early poor ')
print (train_size_ep)
print (test_size_ep)

--MID GOOD
files_mg = {}
files_mg = dir_to_list(mid_good)
index_train_mg,index_test_mg = get_index(index_train_ep,index_test_ep,train_size_ep, test_size_ep)
image_to_t7(mid_good,index_train_mg,index_test_mg)
train_size_mg, test_size_mg = split_dataset(files_mg)
print ('MID GOOD')
print (train_size_mg)
print (test_size_mg)

--MID AVG
files_ma = {}
files_ma = dir_to_list(mid_avg)
index_train_ma,index_test_ma = get_index(index_train_mg,index_test_mg,train_size_mg, test_size_mg)
image_to_t7(mid_avg,index_train_ma,index_test_ma)
train_size_ma, test_size_ma = split_dataset(files_ma)
print ('-MID AVG')
print (#files_ma)
print (train_size_ma)
print (test_size_ma)

--MID POOR
files_mp = {}
files_mp = dir_to_list(mid_poor)
index_train_mp,index_test_mp = get_index(index_train_ma,index_test_ma,train_size_ma, test_size_ma)
image_to_t7_poor(mid_poor, index_train_mp,index_test_mp)
train_size_mp, test_size_mp = split_dataset(files_mp)
print ('mid poor')
print (train_size_mp)
print (test_size_mp)

--LATE GOOD
files_lg = {}
files_lg = dir_to_list(late_good)
index_train_lg,index_test_lg = get_index(index_train_mp,index_test_mp,train_size_mp, test_size_mp)
image_to_t7(late_good, index_train_lg,index_test_lg)
train_size_lg, test_size_lg = split_dataset(files_lg)
print ('LATE GOOD')
print (train_size_lg)
print (test_size_lg)

--LATE AVG
files_la = {}
files_la = dir_to_list(late_avg)
index_train_la,index_test_la= get_index(index_train_lg,index_test_lg,train_size_lg, test_size_lg)
image_to_t7(late_avg,index_train_la,index_test_la)
train_size_la, test_size_la = split_dataset(files_la)
print ('LATE AVG')
print (train_size_la)
print (test_size_la)

--LATE POOR
files_lp = {}
files_lp = dir_to_list(late_poor)
index_train_lp,index_test_lp= get_index(index_train_la,index_test_la,train_size_la, test_size_la)
image_to_t7_poor(late_poor,index_train_lp,index_test_lp)
train_size_lp, test_size_lp = split_dataset(files_lp)
print ('LATE POOR')
print (train_size_lp)
print (test_size_lp)

--save data
torch.save("/home/superuser/project/sugarcane-train.t7", trainData)
torch.save("/home/superuser/project/sugarcane-test.t7", testData)
--trainset = torch.load('/home/superuser/project/sugarcane-train.t7')