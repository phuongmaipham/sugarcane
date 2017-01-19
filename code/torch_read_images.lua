require 'nn';
require 'image';
require 'torchx';
cutorch = require 'cutorch'
--require('trepl')()
--DATASET
early_good="/home/superuser/project/Picture/Early/Good"
early_avg="/home/superuser/project/Picture/Early/Average"
early_poor="/home/superuser/project/Picture/Early/Poor"
mid_good = "home/superuser/project/Picture/Mid/Good"
mid_avg = "home/superuser/project/Picture/Mid/Average"
mid_poor="/home/superuser/project/Picture/Mid/Poor"
late_good = "home/superuser/project/Picture/Late/Good"
late_avg="/home/superuser/project/Picture/Late/Average"
late_poor="/home/superuser/project/Picture/Late/Poor"

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
--      print (files)
--      print (#files)
	return files
end
files = {}
dir_to_list(early_good)
print (files)

--convert images to .t7 file, create train.t7 and test.t7
function image_to_t7(files,index_train,index_test)
	ivch = 3                --#channel
	desImaX = 45            --w
	desImaY = 45            --h
	trainData = {
		data = torch.Tensor(#files, ivch,desImaX,desImaY),
		labels = torch.Tensor(#files),
		size = function() return #files end
		}
	files = {}
	files = dir_to_list(early_good)
	images = {}
	--cutorch.setDevice(1)
	--label1 = {}

	--create a training dataset

	--create a training dataset
	--label = 1 if Good, 2 if Avg and 3 if Poor
	train_size, test_size = split_dataset(files)

	--training data
	for i,file in ipairs(files) do
		if(i<=train_size) then
			--print (i)
			my_image = image.load(file)
			my_image = image.crop(my_image,"c",45,45)
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
			index_train = index_train + 1
		end
	end
	-- testing data
	for i,file in ipairs(files) do
		if(i<=train_size) then
			--print (i)
			my_image = image.load(file)
			my_image = image.crop(my_image,"c",45,45)
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
			index_train = index_train + 1
		end
	end
		--label = torch.Tensor(label1)
		--data = torch.Tensor(#images,3,45,45)
		--print (#images)
		--print (images)

		--for i=1,#images do
		--      data[i] = images[i]
		--      label[i] = label1[i]
		--end
		--Data_to_Write = { data = data, label = label }
	--      image.display (trainData.data[1])
		torch.save("/home/superuser/project/sugarcane-train.t7", trainData)
end

function split_dataset (files)
	--2/3 train, 1/3 test
	total_images = #files
	train_size = math.modf((total_images/3)*2)
	test_size = total_images - train_size
	return  train_size, test_size
end
--start with index = 1
image_to_t7(early_good,1,1)

--image_to_t7(early_avg)
--image_to_t7(early_poor)

--trainset = torch.load('/home/superuser/project/sugarcane-train.t7')
							     [