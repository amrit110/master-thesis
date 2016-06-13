require 'nn'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

-- Function for loading VGG and maniuplating it to
-- output classifier and bounding box heads

backend = 'cudnn'
if backend == 'cudnn' then
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialMaxPooling = cudnn.SpatialMaxPooling
    ReLU = cudnn.ReLU
    SpatialSoftMax = cudnn.SpatialSoftMax
    SpatialFullConvolution = cudnn.SpatialFullConvolution
else
    SpatialConvolution = nn.SpatialConvolution
    SpatialMaxPooling = nn.SpatialMaxPooling
    ReLU = nn.ReLU
    SpatialSoftMax = nn.SpatialSoftMax
    SpatialFullConvolution = nn.SpatialFullConvolution
end

-- Function for loading VGG and maniuplating it to
-- output classifier and bounding box heads
function initializeModel()
    local nOutput = 7
    --local model = torch.load('/mnt/data/pretrainedModels/VGG/VGG.t7')
    local model = loadcaffe.load('deploy.prototxt', '/mnt/data/pretrainedModels/VGG/VGG_ILSVRC_19_layers.caffemodel', 'cudnn')
    for i=#model,28,-1 do
        model.modules[i] = nil
    end
    --model.modules[19] = nn.SpatialMaxPooling(2,2,2,2)
    --model:add(nn.SpatialMaxUnpooling(model.modules[19]))

    -- SPLITTING AND JOINING THE NETWORK:
    local part1 = nn.Sequential()
    for i=19,#model do
        part1:add(model.modules[i]:clone())
    end
    for i=#model,19,-1 do
        model.modules[i] = nil
    end
    --local model = model:clone()
    part1.modules[1] = nn.SpatialMaxPooling(2,2,2,2)
    part1:add(nn.SpatialMaxUnpooling(part1.modules[1]))

    -- Parallel network concatinating feature maps from two different layers
    local parallel = nn.ConcatTable()
    parallel:add(part1)
    parallel:add(nn.Identity())
    model:add(parallel)
    model:add(nn.JoinTable(2))
    model:add(cudnn.SpatialBatchNormalization(768))
    --------------------

    -- Declare classifier head
    local class_head = nn.Sequential()
    class_head:add(SpatialConvolution(768,512,1,1,1,1))
    class_head:add(ReLU(true))
    class_head:add(SpatialConvolution(512,7,1,1,1,1))

    -- Declare box head
    local box_head = nn.Sequential()
    box_head:add(SpatialConvolution(768,512,1,1,1,1))
    box_head:add(ReLU(true))
    box_head:add(SpatialConvolution(512,4,1,1,1,1))

    -- Add heads to model
    local heads = nn.ConcatTable()
    heads:add(class_head)
    heads:add(box_head)
    model:add(heads)
    -- Fix bug by cloning network (?!)
    model2 = model:clone()
    model2:cuda()
    
    return model2
end

function initializeCriterion()
    local criterion = nn.ParallelCriterion()
    criterion:add(cudnn.SpatialCrossEntropyCriterion(opt.classWeights),
        opt.criterionWeights[1])
    criterion:add(nn.MSECriterion(),opt.criterionWeights[2])
    return criterion:cuda()
end



-- Function for saving/loading trained networks
function saveModel(name)
    model:clearState()
    model = model:clone()
    model:float()
    local directoryToSave = 'networks/' .. os.date("%d%m")
    os.execute("mkdir " .. directoryToSave)
    torch.save(directoryToSave .. '/' .. 'model.t7',model)
    torch.save(directoryToSave .. '/' .. 'cost.t7',cost)
    model:cuda()
    collectgarbage()
end

function loadModel(dateSaved)
    local model = torch.load('networks/' .. dateSaved .. '/model.t7')
    cost = torch.load('networks/' .. dateSaved .. '/cost.t7')
    model:cuda()
    collectgarbage()
    return model
end
