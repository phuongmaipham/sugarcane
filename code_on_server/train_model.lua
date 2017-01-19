require 'nn';
require 'image';
require 'torchx';
require 'cunn';

-- get dataset
trainset = torch.load('/home/superuser/project/sugarcane-train.t7')
testset = torch.load('/home/superuser/project/sugarcane-test.t7')

classes = {'good', 'medium', 'bad'}

function trainset:size() 
	return self.data:size(1) 
end
function testset:size() 
	return self.data:size(1) 
end
--Now, to prepare the dataset to be used with StochasticGradient, a couple of things have to be done according to itS documentation: The dataset has to have a size AND The dataset has to have a [i] index operator, so that dataset[i] returns the ith sample.
setmetatable(trainset, 
	{__index = function(t, i) 
					return {
						t.data[i],
						t.labels[i],
					} 
				end}
);
trainset.data = trainset.data:double()
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

net = net:cuda()
trainset.data = trainset.data:cuda()

criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.005
trainer.maxIteration = 5 -- just do 10 epochs of training.
trainer:train(trainset)
torch.save("/home/superuser/project/model1.t7",net:clearState())

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
testset.data = testset.data:cuda()

correct = 0
for i=1,testset:size() do
	local groundtruth = testset.labels[i]
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
print(correct, 100*correct/testset:size().. ' % ')
class_performance = {0, 0, 0}
for i=1,testset:size() do
	local groundtruth = testset.labels[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/testset:size() .. ' %')
end