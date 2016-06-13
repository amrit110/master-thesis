require 'torch'
require 'paths'
require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
--require 'Tiling'

nClasses = 20
-- Load the model
cudnn.fastest = true
cudnn.benchmark = true

SpatialConvolution = cudnn.SpatialConvolution
SpatialFullConvolution = nn.SpatialFullConvolution
SpatialMaxUnpooling = nn.SpatialMaxUnpooling
SpatialBatchNormalization = cudnn.SpatialBatchNormalization
ReLU = cudnn.ReLU

function makeBlock(inputFeatureSize, nBlockType1,nBlockType2)
    local overallBlock = nn.Sequential()
    for i = 1,nBlockType1 do
        local block = nn.Sequential()
        local ConcatBlock = nn.ConcatTable()
        local upconvBlock = nn.Sequential()
        upconvBlock:add(SpatialFullConvolution(inputFeatureSize,inputFeatureSize,3,3,1,1,1,1))
        upconvBlock:add(SpatialBatchNormalization(inputFeatureSize))
        upconvBlock:add(ReLU(true))
        upconvBlock:add(SpatialFullConvolution(inputFeatureSize,inputFeatureSize,3,3,1,1,1,1))
        upconvBlock:add(SpatialBatchNormalization(inputFeatureSize))
        ConcatBlock:add(upconvBlock)
        ConcatBlock:add(nn.Identity())
        block:add(ConcatBlock)
        block:add(nn.CAddTable())
        block:add(ReLU(true))
        overallBlock:add(block)
    end
    for i = 1,nBlockType2 do    
        local block = nn.Sequential()
        local ConcatBlock = nn.ConcatTable()
        local upconvBlock = nn.Sequential()
        upconvBlock:add(SpatialFullConvolution(inputFeatureSize,inputFeatureSize/2,3,3,2,2,1,1))
        upconvBlock:add(SpatialBatchNormalization(inputFeatureSize/2))
        upconvBlock:add(ReLU(true))
        upconvBlock:add(SpatialFullConvolution(inputFeatureSize/2,inputFeatureSize/2,3,3,1,1,1,1))
        upconvBlock:add(SpatialBatchNormalization(inputFeatureSize/2))
        ConcatBlock:add(upconvBlock)
        ConcatBlock:add(SpatialFullConvolution(inputFeatureSize,inputFeatureSize/2,1,1,2,2))
        block:add(ConcatBlock)
        block:add(nn.CAddTable())
        block:add(ReLU(true))
        overallBlock:add(block)
     end
    return overallBlock
end
    
function dilateLayer(inputLayer,dilationFactor,zeroPad)
    local weights = inputLayer.weight
    local bias = inputLayer.bias
    local outputLayer = nn.SpatialDilatedConvolution(inputLayer.nInputPlane,inputLayer.nOutputPlane,inputLayer.kH,inputLayer.kW,1,1,zeroPad,zeroPad,dilationFactor,dilationFactor)
    outputLayer.weight = weights
    outputLayer.bias = bias
    return outputLayer
end


function initialiseModel()
    resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-18.t7')
    -- Remove the fully connected layer
    assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
    resnet:remove(#resnet.modules)
    assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
    resnet:remove(#resnet.modules)
    assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
    resnet:remove(#resnet.modules)
    --resnet.modules[4] = nn.SpatialMaxPooling(2,2,2,2)
    
-- Dilation Stuff 
    resnet:add(SpatialConvolution(512,nClasses,1,1,1,1))
    dilationNet = resnet:clone()
    dilationNet.modules[1].dH = 1
    dilationNet.modules[1].dW = 1
    dilationNet:remove(4)
    dilationNet.modules[4].modules[1].modules[1].modules[1].modules[1] = dilateLayer(resnet.modules[5].modules[1].modules[1].modules[1].modules[1],4,4)
    dilationNet.modules[5].modules[1].modules[1].modules[1].modules[1].dH = 1
    dilationNet.modules[5].modules[1].modules[1].modules[1].modules[1].dW = 1
    dilationNet.modules[5].modules[1].modules[1].modules[1].modules[4] = dilateLayer(resnet.modules[6].modules[1].modules[1].modules[1].modules[4],2,2)
    dilationNet.modules[5].modules[1].modules[1].modules[2].dH = 1
    dilationNet.modules[5].modules[1].modules[1].modules[2].dW = 1
    dilationNet.modules[6].modules[1].modules[1].modules[1].modules[1].dH = 1
    dilationNet.modules[6].modules[1].modules[1].modules[1].modules[1].dW = 1
    dilationNet.modules[6].modules[1].modules[1].modules[1].modules[4] = dilateLayer(resnet.modules[7].modules[1].modules[1].modules[1].modules[4],2,2)
    dilationNet.modules[6].modules[1].modules[1].modules[2].dH = 1
    dilationNet.modules[6].modules[1].modules[1].modules[2].dW = 1
    dilationNet.modules[7].modules[1].modules[1].modules[1].modules[1].dH = 1
    dilationNet.modules[7].modules[1].modules[1].modules[1].modules[1].dW = 1
    dilationNet.modules[7].modules[1].modules[1].modules[1].modules[4] = dilateLayer(resnet.modules[8].modules[1].modules[1].modules[1].modules[4],2,2)
    dilationNet.modules[7].modules[1].modules[1].modules[2].dH = 1
    dilationNet.modules[7].modules[1].modules[1].modules[2].dW = 1

--[[    
    --The upconv net
    local upconvnet = nn.Sequential()
    upconvnet:add(makeBlock(512,1,1))
    upconvnet:add(makeBlock(256,1,1))
    upconvnet:add(makeBlock(128,1,1))
    upconvnet:add(makeBlock(64,2,0))
    upconvnet:add(SpatialMaxUnpooling(resnet.modules[4]))
    upconvnet:add(SpatialBatchNormalization(64))
    upconvnet:add(ReLU(true))
    upconvnet:add(SpatialFullConvolution(64,64,7,7,2,2,3,3,1,1))
    upconvnet:add(SpatialBatchNormalization(64))
    upconvnet:add(ReLU(true))
    upconvnet:add(SpatialFullConvolution(64,nClasses,1,1,1,1))
--]]    
    --The actual full network
    local model = nn.Sequential()
    model:add(dilationNet)
    --model:add(upconvnet)
    return model
end

function saveModel(name)
    model:clearState()
    model = model:clone()
    model:float()
    local directoryToSave = 'networks/' .. os.date("%d%m")
    os.execute("mkdir " .. directoryToSave)
    torch.save(directoryToSave .. '/' .. 'model.t7',model)
    torch.save(directoryToSave .. '/' .. 'cost.t7',cost)
end


function loadModel(dateSaved)
    local model = torch.load('networks/' .. dateSaved .. '/model.t7')
    cost = torch.load('networks/' .. dateSaved .. '/cost.t7')
    collectgarbage()
    return model
end
