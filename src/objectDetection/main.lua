--[[
  This script is used for initializing a new network or loading a pretrained network
  and initializing all the functions to be used for training and inference.
--]]

-- All  global parameters used for data loading, training and inference are stored in opt
opt = {
    offsetCrop = 20,
    nClasses = 5,
    nBatches = 500,
    nValidationBatches = 100,
    batchSize = 10,
    imgSize = {375, 1242},
    cropSize = {224,224},
    maskSize = {56,56},
<<<<<<< HEAD
    criterionWeights = {0,0,1},
=======
    criterionWeights = {0.001,0.01,0},
>>>>>>> 5b6d020372b2442c1eee86e43667b9453168eb85
    --classWeightsClassifier = torch.Tensor{1,0,100,500,500,500,1000},
    --classWeightsClassifier = torch.Tensor{1,0,110,1000,3000},
    classWeightsClassifier = torch.Tensor{1,0,20},
    manualSeed = 10,
    stride = 4,
    nDonkeys = 0,
    sgdState = {
        learningRate = 0.01
    }
}

overallCost = {}
overallCost.training = {}
overallCost.validation = {}

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed = opt.manualSeed

dofile 'model_resnet.lua'
--dofile 'model_dilation.lua'
--model = initialiseModel()
<<<<<<< HEAD
model = loadModel('2605') -- Specify the date the model was saved
=======
model = loadModel('2505') -- Specify the date the model was saved
>>>>>>> 5b6d020372b2442c1eee86e43667b9453168eb85


model:cuda()
criterion = initializeCriterion()
criterion.criterions[1].sizeAverage = false
criterion.criterions[2].sizeAverage = false
dofile 'data.lua'
dofile 'train.lua'       -- Training related functions
dofile 'validation.lua'
dofile 'test.lua'       -- Test related functions

-- TRAIN

--setParameterNetwork('model')
--for i=1,5 do
--    validate()
--    train()
--end
--opt.sgdState.learningRate = 0.001
--for i=1,50 do
--    validate()
--    train()
--end
--
--saveModel()
--os.exit()
