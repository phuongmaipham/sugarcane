require 'nn';
require 'image';
require 'torchx';
--DATASET
train_dir = "/Users/phuongpham/Documents/cane/train"
eval_dir = "/Users/phuongpham/Documents/cane/eval"
--train_dir = "/home/superuser/project/Users/phuongpham/Documents/cane/train"
--eval_dir = "/home/superuser/project/Users/phuongpham/Documents/cane/eval"
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
		my_image:resize(3,32,32)
		table.insert(images, my_image)
		if file:find('good') then
			label1[i] = 1
		else if file:find('medium') then
			label1[i] = 2
		else
			label1[i] = 3
		end
		end
	end
	label = torch.Tensor(label1)
	data = torch.Tensor(#images,3,32,32)
	--print (#images)
	--print (images)
	for i=1,#images do
		data[i] = images[i]
		label[i] = label1[i]
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
--print(#trainset.data)
--print(classes[trainset.label[10]])
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
--print('test size')
--print (testset:size())
print ('=========================TESTSET=========================')
for i=1,testset:size() do
	if testset.label[i]==1 then
		a = 'good'
	else if testset.label[i]==2 then
		a = 'medium'
	else 	
		a = 'bad'
	end
	end
	print ('image', i, 'label', testset.label[i],'-',a)
end
--print ('END')
-- converts the data from a ByteTensor to a DoubleTensor.
trainset.data = trainset.data:double()
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
	mean[i] = trainset.data:select(2, 1):mean() -- mean estimation
	print('Channel ' .. i .. ', Mean: ' .. mean[i])
	trainset.data:select(2, 1):add(-mean[i]) -- mean subtraction
	
	stdv[i] = trainset.data:select(2, i):std() -- std estimation
	print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
	trainset.data:select(2, i):div(stdv[i]) -- std scaling
end
--print(trainset:size()) -- just to test
-- remember: our dataset is #samples x #channels x #height x #width
net = nn.Sequential()

--MODEL
--[[
net:add(nn.SpatialConvolution(3, 32, 6, 6))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(32, 64, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(64, 128, 3, 3))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(128, 128, 3, 3))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.View(128*2*2))
net:add(nn.Threshold())
net:add(nn.Linear(128*2*2, 2048))
net:add(nn.Threshold())
net:add(nn.Linear(2048, 2048))
net:add(nn.Threshold())
net:add(nn.Linear(2048, 3))
--]]

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Threshold())

net:add(nn.View(16*5*5))

net:add(nn.Linear(16*5*5, 120))
net:add(nn.Threshold())
net:add(nn.Linear(120, 84))
net:add(nn.Threshold())
net:add(nn.Linear(84, 3))

criterion = nn.CrossEntropyCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.005
trainer.maxIteration = 5 -- just do 10 epochs of training.
trainer:train(trainset)
--print('TESTSET IMAGE')
--print(testset.label[3])
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
--itorch.image(testset.data[3])
--print(classes[testset.label[3]])
--print ('PREDICTED CLASSES')
--print(predicted)
--for i=1,predicted:size(1) do
--	print(classes[i], predicted[i])
--end
correct = 0
for i=1,testset:size() do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	print (i)
	for i=1,prediction:size(1) do
		print('lables ', classes[i], prediction[i])

	end
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	--print ('INDICES')
	--print (indices[1])
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print(correct, 100*correct/20 .. ' % ')
class_performance = {0, 0, 0}
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