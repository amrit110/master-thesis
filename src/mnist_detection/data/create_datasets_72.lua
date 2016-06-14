require 'dataset-mnist'
require 'image'
--[[
This scipt construct 72x72 images with 2-5 MNIST digits.
The digits are non-overlaping and are scaled from their original
size of 32x32 to 16x16. The labels are stored as
[class, x1, y1, x2, y2],
where the bounding box coordinates taken from the original placing
of the 16x16 MNIST digit but then shrunk to 8x8.
The dataset is normalized and stored as 'mnist_72.t7'.
--]]

-- Parameters
n_channels = 1
mnist_size = 32
mnist_scale = 16
img_size = 72 -- MULTIPLE OF 4
mask_size = 18
shrinkage = 4

-- Number of samples to create
n_training = 10000
n_test = 1000
-- Number of digits to load from original MNIST data set
n_mnist_digits_train = 40000
n_mnist_digits_test = 5000

-- Data import
geometry = {mnist_size, mnist_size}
trainData = mnist.loadTrainSet(n_mnist_digits_train, geometry)
testData = mnist.loadTestSet(n_mnist_digits_test, geometry)


-- Function to evaluate if coordinates p & q are valid and result in no overlap
function valid_coord(p,q)
    for i=1,#p:totable()-1 do
        for j=i+1,#p:totable() do
            if math.abs(q[i]-q[j]) < mnist_scale and math.abs(p[i]-p[j]) < mnist_scale then
                return false
            end
        end
    end
    return true
end

-- Main function to create 1 sample with labels from 'data_set'
function expand(data_set)
    local N = data_set.data:size(1)
    local img = torch.zeros(img_size,img_size)
    local label_old = torch.Tensor(img_size,img_size):fill(11)

    -- between 2 and 5 digits in each image
    local n_digits = math.random(2,5)
    local label_new = torch.Tensor(n_digits, 5)
    local p = torch.Tensor(n_digits)
    local q = torch.Tensor(n_digits)
    -- sample a set of valid coordinates for the digits
    repeat
        for i=1,n_digits do
            p[i] = math.random(1,img_size-mnist_scale-1)
            q[i] = math.random(1,img_size-mnist_scale-1)
        end
    until valid_coord(p,q)

    -- place the digits
    for i=1,n_digits do
        local indx = math.random(1,N)
        local digit = image.scale(data_set.data[indx], mnist_scale)
        img[{{p[i],p[i]+mnist_scale-1},{q[i],q[i]+mnist_scale-1}}] = digit

        local label = data_set.labels[indx]
        --pl = torch.ceil((p[i]+shrinkage)/4)
        --ql = torch.ceil((q[i]+shrinkage)/4)
        label_old[{{p[i],p[i]+mnist_scale-1}, {q[i],q[i]+mnist_scale-1}}] = label
        local y1 = p[i] + shrinkage
        local x1 = q[i] + shrinkage
        local y2 = p[i] + mnist_scale - shrinkage
        local x2 = q[i] + mnist_scale - shrinkage
        label_new[i] = torch.Tensor{label, x1, y1, x2, y2}

    end

    -- add noise
    img:add(torch.randn(img_size,img_size)*5)

    return img, label_new, label_old
end



-- New containers
trainData_new = torch.Tensor(n_training, 1, img_size, img_size)
trainLabels_old = torch.Tensor(n_training, img_size, img_size)
trainLabels_new = {}
testData_new = torch.Tensor(n_test, 1, img_size, img_size)
testLabels_old = torch.Tensor(n_test, img_size, img_size)
testLabels_new = {}

-- Create training set
for i=1,n_training do
    local img, label_new, label_old = expand(trainData)
    trainData_new[i] = img
    trainLabels_new[i] = label_new
    trainLabels_old[i] = label_old
end

-- create test set
for i=1,n_test do
    local img, label_new, label_old = expand(testData)
    testData_new[i] = img
    testLabels_new[i] = label_new
    testLabels_old[i] = label_old
end

-- Store all in one table
training = {}
training.data = trainData_new
training.label = trainLabels_new
training.label_old = trainLabels_old
test = {}
test.data = testData_new
test.label = testLabels_new
test.label_old = testLabels_old

-- Normalize it, zero mean unit variance with respect to training data
mean = training.data:mean()
training.data:add(-mean)
test.data:add(-mean)
stdv = training.data:std()
training.data:div(stdv)
test.data:div(stdv)
torch.save('mnist_72.t7', training)
torch.save('mnist_72_test.t7', test)

