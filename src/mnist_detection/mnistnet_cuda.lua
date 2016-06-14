require 'torch'
require 'nn'

function createModel()
    backend = 'cudnn'
    if backend == 'cudnn' then
        SpatialConvolution = cudnn.SpatialConvolution
        SpatialMaxPooling = cudnn.SpatialMaxPooling
        ReLU = cudnn.ReLU
        SpatialBatchNormalization = cudnn.SpatialBatchNormalization
    else
        SpatialConvolution = nn.SpatialConvolutionMM
        SpatialMaxPooling = nn.SpatialMaxPooling
        ReLU = nn.ReLU
        SpatialBatchNormalization = nn.SpatialBatchNormalization
    end
    
    --feature extractor part
    local featurenet = nn.Sequential()
    featurenet = nn.Sequential()
    featurenet:add(SpatialConvolution(1,32,7,7,1,1,0,0))
    --featurenet:add(SpatialBatchNormalization(32))
    featurenet:add(ReLU(true))
    featurenet:add(SpatialMaxPooling(2,2,2,2))
    featurenet:add(SpatialConvolution(32,64,5,5,1,1,0,0))
    --featurenet:add(SpatialBatchNormalization(64))
    featurenet:add(ReLU(true))
    featurenet:add(SpatialMaxPooling(2,2,2,2))
    featurenet:add(SpatialConvolution(64,128,3,3,1,1,0,0))
    --featurenet:add(SpatialBatchNormalization(128))
    featurenet:add(ReLU(true))
    
    --classifier part (without spatial soft max)
    local classifier = nn.Sequential()
    classifier:add(SpatialConvolution(128,256,1,1))
    --classifier:add(SpatialBatchNormalization(256))
    classifier:add(ReLU(true))
    classifier:add(SpatialConvolution(256,512,1,1))
    --classifier:add(SpatialBatchNormalization(512))
    classifier:add(ReLU(true))
    classifier:add(SpatialConvolution(512,11,1,1))

    --bounding box regressor part
    local regressor = nn.Sequential()
    regressor:add(SpatialConvolution(128,256,1,1))
    --regressor:add(SpatialBatchNormalization(256))
    regressor:add(ReLU(true))
    regressor:add(SpatialConvolution(256,512,1,1))
    --regressor:add(SpatialBatchNormalization(512))
    regressor:add(ReLU(true))
    regressor:add(SpatialConvolution(512,1024,1,1))
    --regressor:add(SpatialBatchNormalization(1024))
    regressor:add(ReLU(true))
    regressor:add(SpatialConvolution(1024,4,1,1))
    --regressor:add(nn.ReLU(true))
    
    --head of the network
    local heads = nn.ConcatTable():add(classifier):add(regressor)
    
    --final model
    local model = nn.Sequential():add(featurenet):add(heads)

    --return featurenet, classifier, regressor, model
    if backend == 'cudnn' then
        model:cuda()
    end
    return model
end

function saveModel()
    model:clearState()
    model:float()
    torch.save('trained/model.t7', model)
    torch.save('trained/cost.t7', cost)
    torch.save('trained/error_train.t7', error_train)
    torch.save('trained/error_valid.t7', error_valid)
    torch.save('trained/opt.t7', opt)
end

function loadModel()
    model = torch.load('trained/model.t7')
    cost = torch.load('trained/cost.t7')
    error_train = torch.load('trained/error_train.t7')
    error_valid = torch.load('trained/error_valid.t7')
    opt = torch.load('trained/opt.t7')
end

--function saveMat()
--    local matio = require 'matio'
--    matio.save('matfiles/error_train.mat',torch.Tensor(error_train))
--    matio.save('matfiles/error_valid.mat',torch.Tensor(error_valid))
--    matio.save('matfiles/cost_train_class.mat',torch.Tensor(cost.train.class))
--    matio.save('matfiles/cost_train_box.mat',torch.Tensor(cost.train.box))
--    matio.save('matfiles/cost_val_class.mat',torch.Tensor(cost.val.class))
--    matio.save('matfiles/cost_val_box.mat',torch.Tensor(cost.val.box))
--end

