--script used to create the model for semantic segmentation--

require 'torch'
require 'paths'
require 'cudnn'
require 'nn'
require 'cunn'
require 'image'

-- Load the original VGG
modelPath = '/mnt/data/pretrainedModels/'

local convnetLoad = torch.load(modelPath .. 'VGG/VGG.t7') 

--the following part can be modified to either be the full network or
--lightweight
local weights1 = convnetLoad.modules[39].weight
local bias1 = convnetLoad.modules[39].bias
local weights2 = convnetLoad.modules[42].weight
local bias2 = convnetLoad.modules[42].bias


for i = 46,38,-1 do
    convnetLoad.modules[i] = nil
end

convnetLoad:add(cudnn.SpatialConvolution(512,4096,7,7,1,1))
convnetLoad:add(cudnn.ReLU(true))
convnetLoad:add(cudnn.SpatialConvolution(4096,4096,1,1,1,1))
convnetLoad:add(cudnn.ReLU(true))

convnetLoad.modules[38].weight = weights1:clone()
convnetLoad.modules[38].bias = bias1:clone()
convnetLoad.modules[40].weight = weights2:clone()
convnetLoad.modules[40].bias = bias2:clone()



--convnetLoad.modules[46] = nil
--convnetLoad.modules[45] = nil
convnetLoad.modules[37] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[28] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[19] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[10] = nn.SpatialMaxPooling(2,2,2,2)
convnetLoad.modules[5] = nn.SpatialMaxPooling(2,2,2,2)

SpatialConvolution = cudnn.SpatialConvolution
SpatialFullConvolution = cudnn.SpatialFullConvolution
convnet = convnetLoad:clone()

--the deconvolution network--
upsamplingnet = nn.Sequential()
upsamplingnet:add(cudnn.SpatialBatchNormalization(4096))
upsamplingnet:add(SpatialFullConvolution(4096,512,7,7,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(nn.SpatialMaxUnpooling(convnet.modules[37]))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(nn.SpatialMaxUnpooling(convnet.modules[28]))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,512,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(512))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(512,256,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(256))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(nn.SpatialMaxUnpooling(convnet.modules[19]))
upsamplingnet:add(SpatialFullConvolution(256,256,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(256))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(256,256,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(256))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(256,256,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(256))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(256,128,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(128))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(nn.SpatialMaxUnpooling(convnet.modules[10]))
upsamplingnet:add(SpatialFullConvolution(128,128,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(128))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialFullConvolution(128,64,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(64))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(nn.SpatialMaxUnpooling(convnet.modules[5]))
upsamplingnet:add(SpatialFullConvolution(64,64,3,3,1,1,1,1))
upsamplingnet:add(cudnn.SpatialBatchNormalization(64))
upsamplingnet:add(cudnn.ReLU(true))
upsamplingnet:add(SpatialConvolution(64,opt.nClasses,1,1,1,1))

model = nn.Sequential() 
model:add(convnet)
model:add(upsamplingnet)
model = model:clone():cuda()

