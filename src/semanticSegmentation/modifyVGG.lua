--script to modify a pretrained VGG by removing last layers and saving it so
--that it can be used for semantic segmentation

require 'torch'
require 'paths'
require 'cudnn'
require 'nn'
require 'cunn'
require 'image'

-- Load the model
modelPath = '/mnt/data/pretrainedModels/'
local convnetLoad = torch.load(modelPath .. 'VGG/VGG.t7') 

--weights and biases for copying to transforming fc layers to fully
--convolutional
local weights1 = convnetLoad.modules[39].weight
local bias1 = convnetLoad.modules[39].bias
local weights2 = convnetLoad.modules[42].weight
local bias2 = convnetLoad.modules[42].bias

--deleting the last layers
for i = 46,38,-1 do
    convnetLoad.modules[i] = nil
end

--converting and appending
convnetLoad:add(cudnn.SpatialConvolution(512,4096,7,7,1,1))
convnetLoad:add(cudnn.ReLU(true))
convnetLoad:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
convnetLoad:add(cudnn.ReLU(true))

convnetLoad.modules[38].weight = weights1:clone()
convnetLoad.modules[38].bias = bias1:clone()
convnetLoad.modules[40].weight = weights2:clone()
convnetLoad.modules[40].bias = bias2:clone()


--converting cudnn to nn pooling layers so that unpooling layers can be used
convnetLoad.modules[37] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[28] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[19] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[10] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[5] = nn.SpatialMaxPooling(2,2,2,2)

convnet = convnetLoad:clone()

-- convnet can now be used for classification, segmentation
