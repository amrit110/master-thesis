require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')
cudnn.fastest = true
cudnn.benchmark = true

resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-18.t7')
function setDilation(index, zeroPadding, dilation)
    local subnet = resnet
    local lastIndex = index[#index]
    index[#index] = nil
    for i, v in pairs(index) do
        subnet = subnet.modules[v]
    end
    local inModule = subnet.modules[lastIndex]

    local nInputPlane = inModule.nInputPlane
    local nOutputPlane = inModule.nOutputPlane
    local weights = inModule.weight
    local bias = inModule.bias
    local outModule = nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane,3,3,1,1,zeroPadding,zeroPadding, dilation, dilation)
    outModule.weight = weights:clone()
    outModule.bias = bias:clone()

    subnet:remove(lastIndex)
    subnet:insert(outModule, lastIndex)
end

function removeStride(index,subnet)
    local subnet = resnet
    for i, v in pairs(index) do
        subnet = subnet.modules[v]
    end
    local inModule = subnet

    inModule.dW = 1
    inModule.dH = 1
end

function makeBBHead()
    local box_head = nn.Sequential()
    box_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
    box_head:add(cudnn.SpatialBatchNormalization(1024)) 
    box_head:add(cudnn.ReLU(true))
    box_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
    box_head:add(cudnn.SpatialBatchNormalization(1024)) 
    box_head:add(cudnn.ReLU(true))
    box_head:add(cudnn.SpatialConvolution(1024,4,1,1,1,1))
    return box_head
end


--removeStride({1})
--setDilation({5,1,1,1,1}, 4, 4)
removeStride({6,1,1,1,1})
setDilation({6,1,1,1,4}, 2, 2)
removeStride({6,1,1,2})
setDilation({6,2,1,1,1}, 2, 2)
setDilation({6,2,1,1,4}, 2, 2)
setDilation({7,1,1,1,1}, 2, 2)
setDilation({7,2,1,1,1}, 4, 4)
setDilation({7,2,1,1,4}, 4, 4)
setDilation({8,1,1,1,1}, 4, 4)
setDilation({8,2,1,1,1}, 8, 8)
setDilation({8,2,1,1,4}, 8, 8)


removeStride({7,1,1,1,1})
setDilation({7,1,1,1,4}, 4, 4)
removeStride({7,1,1,2})

removeStride({8,1,1,1,1})
setDilation({8,1,1,1,4}, 8, 8)
removeStride({8,1,1,2})

-- Remove 7x7 avg pooling, view, linear
resnet:remove(11)
resnet:remove(10)
resnet:remove(9)

-- Declare classifier head
local class_head = nn.Sequential()
class_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
class_head:add(cudnn.SpatialBatchNormalization(1024)) 
class_head:add(cudnn.ReLU(true))
class_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
class_head:add(cudnn.SpatialBatchNormalization(1024)) 
class_head:add(cudnn.ReLU(true))
class_head:add(cudnn.SpatialConvolution(1024,opt.nClasses,1,1,1,1))
class_head:add(nn.SpatialSoftMax())
--Declare box head
local box_head = makeBBHead()

--Declare range head
local range_head = nn.Sequential()
--range_head:add(cudnn.SpatialConvolution(512,1024,1,1,1,1))
--range_head:add(cudnn.SpatialBatchNormalization(1024)) 
--range_head:add(cudnn.ReLU(true))
--range_head:add(cudnn.SpatialConvolution(1024,1024,1,1,1,1))
--range_head:add(cudnn.SpatialBatchNormalization(1024)) 
--range_head:add(cudnn.ReLU(true))
range_head:add(cudnn.SpatialConvolution(512,1,1,1,1))

-- Combine all heads
local heads = nn.ConcatTable()
heads:add(class_head)
heads:add(box_head)
--heads:add(range_head)

--local heads = nn.ConcatTable()
--heads:add(cudnn.SpatialConvolution(512,opt.nClasses,1,1))
--heads:add(cudnn.SpatialConvolution(512,4,1,1))
--heads:add(cudnn.SpatialConvolution(512,1,1,1))

model = nn.Sequential()
model:add(resnet)
model:add(heads)

model:cuda()

function initializeCriterion()
    local criterion = nn.ParallelCriterion()
    -- Add criterion for classifier, box regressor and range
    criterion:add(nn.MSECriterion(),opt.criterionWeights[1])
    criterion:add(nn.MSECriterion(),opt.criterionWeights[2])
    --criterion:add(nn.MSECriterion(),opt.criterionWeights[3])
    return criterion:cuda()
end


function saveModel()
    -- Clear network of intermediate activation functions
    model:clearState()
    -- Transfer network to CPU
    model = model:clone()
    model:float()
    -- Store network and cost based on date
    local directoryToSave = 'networks/' .. os.date("%d%m")
    os.execute("mkdir " .. directoryToSave)
    torch.save(directoryToSave .. '/' .. 'model.t7',model)
    torch.save(directoryToSave .. '/' .. 'cost.t7',cost)
end


function loadModel(dateSaved)
    -- Takes argument of string 'dataSaved' eg '0411'
    local model = torch.load('networks/' .. dateSaved .. '/model.t7')
    cost = torch.load('networks/' .. dateSaved .. '/cost.t7')
    collectgarbage()
    return model
end
