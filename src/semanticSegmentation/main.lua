-- the main script for semantic segmentation--


--Requires--
require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'gnuplot'
require 'cutorch'


torch.setdefaulttensortype('torch.FloatTensor')
cudnn.benchmark = true
cudnn.fastest = true

--parameters--
opt = {
    batchSize = 2,
    nTrainingBatches = 10000,
    nValidationBatches = 50,
    sgdState = {
        learningRate = 0.002
    },
    loadPrevious = true,
    nClasses = 20,
    classWeights = torch.Tensor{0,1,6,2,55,47,31,185,70,2,36,9,35,300,6,212,480,128,371,108}
}

--loading a pre-trained model from disk--
if opt.loadPrevious then
    torch.setdefaulttensortype('torch.FloatTensor')
    model = torch.load('/home/amrkri/Master_Thesis/savedModels/semanticModel/model.t7')
    model = model:cuda()
else
    --creating a model with the VGG architecture and deconvnet--
    dofile 'model.lua'
    --creating a model using resnet-18 and dilating it--
    --[[dofile 'model_dilation.lua'
    model = initialiseModel():cuda()--]]
end


--load data related fucntions--
dofile 'data.lua'

--load the training and validation script--
dofile 'train.lua'

--train the entire model or only the deconvnet--
setParametersNet('model')


