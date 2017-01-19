require 'nn';
require 'image';
require 'torchx';

-- get dataset
trainset = torch.load('/home/superuser/project/sugarcane-train.t7')
testset = torch.load('/home/superuser/project/sugarcane-test.t7')

function trainset:size() 
	return self.data:size(1) 
end
function testset:size() 
	return self.data:size(1) 
end
-- display some images and labels
print ('=========================TRAINSET=========================')
for i,10 do
	i = math.random(1,3036)
	if testset.labels[i]==1 then
		l = 'good'
	else if testset.labels[i]==2 then
		l = 'medium'
	else 
		l = 'bad'
	end
	end
	print ('image', i, 'label', testset.labels[i],'-',l)
end
print ('=========================TESTSET=========================')
for i,10 do
        i = math.random(1,1526)
        if testset.labels[i]==1 then
                l = 'good'
        else if testset.labels[i]==2 then
                l = 'medium'    
        else
                l = 'bad'
        end
        end
        print ('image', i, 'label', testset.labels[i],'-',l)
end

-- converts the data from a ByteTensor to a DoubleTensor.
trainset.data = trainset.data:double()

-- choose a random number
function randomFloat(lower, greater)
    return lower + math.random()  * (greater - lower);
end

--brightness and contrast adjustment 
function bright_contrast (im)
	a = randomFloat(-0.9, 0.9)
	b = randomFloat(-2.5, 2.5)
	for i=1,3 do -- over each image channel
		im:select(2, i):mul(b):add(a)
	end
end
-- converts the data from a ByteTensor to a DoubleTensor.
trainset.data = trainset.data:double()

-- data augmentation with possibility = 0.5
for i,#trainset.data do
	a = math.random(1,10)
	if (a < 6) do 
		bright_contrast(trainset.data[i])
	end
end