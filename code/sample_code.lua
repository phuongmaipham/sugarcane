#!/usr/bin/python

require 'nn';
require 'image';
require 'torchx';
--DATASET
train_dir = "/Users/phuongpham/Documents/cane/train"
eval_dir = "/Users/phuongpham/Documents/cane/eval"

-- add: data augumentations
-- view points
-- change: number of layer

function image_to_t7(dir, train)
	label = {}
	files = {}
	--create a list of images in a directory 
	for file in paths.files(dir) do
		if (file:find('jpg') or file:find('JPG') or file:find('png') or file:find('PNG') or file:find('jpeg') or file:find('JPEG')) then
		table.insert(files,paths.concat(dir,file))
		end
	end
	images = {}
	label1 = {}
	for i,file in ipairs(files) do
		my_image = image.load(file)
		my_image:resize(3,16,16)
		table.insert(images, my_image)
		if file:find('good') then
			label1[i] = 3
		else if file:find('medium') then
			label1[i] = 2
		else
			label1[i] = 1
		end
		end
	end
	label = torch.Tensor(label1)
	data = torch.Tensor(#images,3,16,16)
	print (#images)
	print (images)
	for i=1,#images do
		data[i] = images[i]
	end
	Data_to_Write = { data = data, label = label }
	if train == 1 then
		torch.save("/Users/phuongpham/Documents/sugarcane-train.t7", Data_to_Write)
	else
		torch.save("/Users/phuongpham/Documents/sugarcane-eval.t7", Data_to_Write)
	end
end
train = 1
image_to_t7(train_dir,train)
train = 0
image_to_t7(eval_dir,train)
trainset = torch.load('/Users/phuongpham/Documents/sugarcane-train.t7')
testset = torch.load('/Users/phuongpham/Documents/sugarcane-eval.t7')
classes = {'good', 'medium', 'bad'}
print(trainset)
print(#trainset.data)
print(classes[trainset.label[10]])
--Now, to prepare the dataset to be used with StochasticGradient, a couple of things have to be done according to itS documentation: The dataset has to have a size AND The dataset has to have a [i] index operator, so that dataset[i] returns the ith sample.
setmetatable(trainset, 
	{__index = function(t, i) 
					return {
						t.data[i],
						t.label[i],
					} 
				end}
);

function trainset:size() 
	return self.data:size(1) 
end
function testset:size() 
	return self.data:size(1) 
end
print('test size')
print (testset:size())
print ('TEST')
for i=1,trainset:size() do
	print (trainset.label[i])
end
print ('END')
-- converts the data from a ByteTensor to a DoubleTensor.
trainset.data = trainset.data:double()
print(trainset:size()) -- just to test
-- remember: our dataset is #samples x #channels x #height x #width
net = nn.Sequential()

--MODEL
net:add(nn.SpatialConvolution(3, 6, 2, 2))
net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.SpatialConvolution(6, 9, 2, 2))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(9, 16, 2, 2))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.View(16*1*1))
net:add(nn.Linear(16*1*1, 120))
net:add(nn.Threshold())
net:add(nn.Linear(120, 84))
net:add(nn.Threshold())
net:add(nn.Linear(84, 12))
net:add(nn.Threshold())
net:add(nn.Linear(12, 3))

criterion = nn.CrossEntropyCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 200 -- just do 5 epochs of training.
trainer:train(trainset)
print('TESTSET IMAGE')
print(testset.label[3])
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
--itorch.image(testset.data[3])
print(classes[testset.label[3]])
predicted = net:forward(testset.data[3])
print ('PREDICTED CLASSES')
print(predicted)
for i=1,predicted:size(1) do
	print(classes[i], predicted[i])
end
correct = 0
for i=1,testset:size() do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print(correct, 100*correct/20 .. ' % ')
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,testset:size() do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/20 .. ' %')
end