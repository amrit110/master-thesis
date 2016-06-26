--[[
  This file contains functions for initialising, loading and saving the network.
  The network uses a pretrained version of the resnet34 as feature extrator.
--]]

require 'paths'
require 'nn'
require 'cudnn'
require 'cunn'
require 'image'
require 'tiling'

cudnn.fastest = true
cudnn.benchmark = true

function initialiseModel()
    -- Load complete resnet34
    local resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-34.t7')

    -- Remove the fully connected layer
    assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
    resnet:remove(#resnet.modules)
    assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
    resnet:remove(#resnet.modules)
    assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
    resnet:remove(#resnet.modules)
    
    -- Declare classifier head
    local class_head = nn.Sequential()
    class_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
    class_head:add(cudnn.SpatialBatchNormalization(1024)) 
    class_head:add(cudnn.ReLU(true))
    class_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
    class_head:add(cudnn.SpatialBatchNormalization(1024)) 
    class_head:add(cudnn.ReLU(true))
    class_head:add(cudnn.SpatialConvolution(1024,opt.nClasses*((32/opt.stride)^2),1,1,1,1))
    class_head:add(nn.Tiling(32/opt.stride))
    class_head:add(nn.SpatialSoftMax())
    --Declare box head
    local box_head = makeBBHead()
    --Declare range head
    local range_head = nn.Sequential()
    range_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
    range_head:add(cudnn.SpatialBatchNormalization(1024)) 
    range_head:add(cudnn.ReLU(true))
    range_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
    range_head:add(cudnn.SpatialBatchNormalization(1024)) 
    range_head:add(cudnn.ReLU(true))
    range_head:add(cudnn.SpatialConvolution(1024,1*((32/opt.stride)^2),1,1,1,1))
    range_head:add(nn.Tiling(32/opt.stride))
  
    -- Combine all heads
    local heads = nn.ConcatTable()
    heads:add(class_head)
    heads:add(box_head)
    heads:add(range_head)

    -- Add breakaway part and heads to model
    local model = nn.Sequential()
    model:add(resnet)
    model:add(heads)
    return model
end


function initializeCriterion()
    local criterion = nn.ParallelCriterion()
    -- Add criterion for classifier, box regressor and range
    criterion:add(nn.MSECriterion(),opt.criterionWeights[1])
    criterion:add(nn.SmoothL1Criterion(),opt.criterionWeights[2])
    --criterion:add(nn.MSECriterion(),opt.criterionWeights[2])
    criterion:add(nn.MSECriterion(),opt.criterionWeights[3])
    return criterion:cuda()
end

function makeBBHead()
    local box_head = nn.Sequential()
    box_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
    box_head:add(cudnn.SpatialBatchNormalization(1024)) 
    box_head:add(cudnn.ReLU(true))
    box_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
    box_head:add(cudnn.SpatialBatchNormalization(1024)) 
    box_head:add(cudnn.ReLU(true))
    box_head:add(cudnn.SpatialConvolution(1024,4*((32/opt.stride)^2),1,1,1,1))
    box_head:add(nn.Tiling(32/opt.stride))
    return box_head
end



function saveModel()
    -- Clear network of intermediate activation functions
    model:clearState()
    -- Transfer network to CPU
    model = model:clone()
    model:float()
    -- Store network and cost based on date
    local directoryToSave = '/home/amrkri/Github/detection_residual/networks/' .. os.date("%d%m")
    os.execute("mkdir " .. directoryToSave)
    torch.save(directoryToSave .. '/' .. 'model.t7',model)
    torch.save(directoryToSave .. '/' .. 'cost.t7',cost)
    --torch.save(directoryToSave .. '/' .. 'overallCost.t7',overallCost)
end


function loadModel(dateSaved)
    -- Takes argument of string 'dataSaved' eg '0411'
    local model = torch.load('/home/amrkri/Github/detection_residual/networks/' .. dateSaved .. '/model.t7')
    collectgarbage()
    return model
end
