require 'torch'
require 'nn'
require 'optim'
require 'mnistnet_cuda'
dofile 'plotfunc_cuda.lua'
torch.setdefaulttensortype('torch.FloatTensor')
require 'cudnn'
require 'cunn'
require 'cutorch'


-- Parameters and options
-- Specify absolut path to this folder:
path = '/home/jonlar/Master-Thesis/src/mnist_detection/'
pretrained = true
geometry = {26,26}


-- Construct network and options
if pretrained == false then
    opt = {
        epoch = 0,
        batch_size = 50,
        training_size = 26,
        criterionWeights = {1,0.1},
        classWeights = torch.ones(11),
        sgdState = {
            learningRate = 0.1
        }
    }
    -- Class weight for background class:
    opt.classWeights[11] = 1
    model = createModel()
    cost = {
        train = {
            class ={},
            box ={}
        },
        val = {
            class ={},
            box ={}
        }
    }
    error_train = {}
    error_valid = {}
else
    loadModel()
    model:cuda()
end

-- Declare cost functions
classes = {'0','1','2','3','4','5','6','7','8','9','Background'}
criterion_class = nn.CrossEntropyCriterion(opt.classWeights)
criterion_bb = nn.SmoothL1Criterion()
criterion = nn.ParallelCriterion()
criterion:add(criterion_class,opt.criterionWeights[1])
criterion:add(criterion_bb,opt.criterionWeights[2])
confusion = optim.ConfusionMatrix(classes)


-- Import data sets, split training and validation
valSize = 5000
trainSize = 45000
trainset = torch.load(path .. 'data/mnist_train.t7')
valset = {}
valset.data = trainset.data[{{trainSize+1,trainSize+valSize}}]
valset.label = trainset.label[{{trainSize+1,trainSize+valSize}}]
function valset:size()
    return valset.label:size(1)
end
trainset.data = trainset.data[{{1,trainSize}}]
trainset.label = trainset.label[{{1,trainSize}}]
testset = torch.load(path .. 'data/mnist_test.t7')
n_training = trainset:size()

-- Training function
function train()
    opt.epoch = opt.epoch + 1
    local cost_box_tmp = 0
    local cost_class_tmp = 0
    local parameters, gradParameters = model:getParameters()

    criterion:cuda()
    local time = sys.clock()
    for t = 1,trainset:size(),opt.batch_size do
        -- Prepare mini batch:
        local inputs = torch.CudaTensor(opt.batch_size,1,opt.training_size,opt.training_size)
        local targets_class = torch.CudaTensor(opt.batch_size)
        local targets_bb = torch.CudaTensor(opt.batch_size,4,1,1)
        local targets_bb_mask = torch.CudaTensor(opt.batch_size):fill(1)
        local k = 1
        --for i = t,math.min(t+opt.batch_size-1,trainset:size()) do
        for i = 1,opt.batch_size do
            local index = torch.random(trainset:size())
            local input = trainset.data[index]:cuda()
            local target_class = trainset.label[index][1]
            local target_bb = trainset.label[index][{{2,5}}]:cuda()
            --target = target:squeeze()
            if target_class == 11 then
                targets_bb_mask[k] = 0
                target_bb = target_bb:fill(0)
            end
            inputs[k] = input
            targets_class[k] = target_class
            targets_bb[k] = target_bb
            k = k + 1
        end

        -- Evaluate cost and gradients
        local feval = function(x)
            collectgarbage()
            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()
            local outputs = model:forward(inputs)
            -- Supress bb output with class background:
            for i=1,opt.batch_size do
                outputs[2][i] = outputs[2][i]:mul(targets_bb_mask[i])
            end
            local f = criterion:forward(outputs, {targets_class, targets_bb})
            cost_class_tmp = cost_class_tmp + criterion.criterions[1].output
            cost_box_tmp = cost_box_tmp + criterion.criterions[2].output
            local df_do = criterion:backward(outputs, {targets_class, targets_bb})
            model:backward(inputs, df_do)

            -- update confusion
            for i=1,opt.batch_size do
                confusion:add(outputs[1][i]:squeeze(), targets_class[i])
            end

            return f,gradParameters
        end

        -- Perform SGD
        optim.sgd(feval, parameters, opt.sgdState)
        -- display progress
        xlua.progress(t, trainset:size())
    end

    -- Store average cost from epoch
    cost_class_tmp = cost_class_tmp/trainset:size()
    cost_box_tmp = cost_box_tmp/trainset:size()
    table.insert(cost.train.class, cost_class_tmp)
    table.insert(cost.train.box, cost_box_tmp)
    confusion:updateValids()
    table.insert(error_train, confusion.totalValid)
    print('<trainer> trainingset accuracy: ' .. confusion.totalValid)
    confusion:zero()
end


-- Evaluation function
function evaluateError(datasetString)
    local cost_box_tmp = 0
    local cost_class_tmp = 0
    local dataset
    if datasetString == 'validation' then
        dataset = valset
    elseif datasetString == 'test' then
        dataset = testset
    else
        assert('Unknown dataset!')
    end
    criterion:cuda()

    local time = sys.clock()
    for t = 1,dataset:size(),opt.batch_size do
        -- Prepare mini batch:
        local inputs = torch.CudaTensor(opt.batch_size,1,opt.training_size,opt.training_size)
        local targets_class = torch.CudaTensor(opt.batch_size)
        local targets_bb = torch.CudaTensor(opt.batch_size,4,1,1)
        local targets_bb_mask = torch.CudaTensor(opt.batch_size):fill(1)
        local k = 1
        for i = t,math.min(t+opt.batch_size-1,dataset:size()) do
            -- removed cuda from this:
            --local index = torch.random(dataset:size())
            local input = dataset.data[i]:cuda()
            local target_class = dataset.label[i][1]
            local target_bb = dataset.label[i][{{2,5}}]:cuda()
            --target = target:squeeze()
            if target_class == 11 then
                targets_bb_mask[k] = 0
                target_bb = target_bb:fill(0)
            end
            inputs[k] = input
            targets_class[k] = target_class
            targets_bb[k] = target_bb
            k = k + 1
        end

        -- Evaluate cost and error
        collectgarbage()
        local outputs = model:forward(inputs)
        -- Supress bb output with class background:
        for i=1,opt.batch_size do
            outputs[2][i] = outputs[2][i]:mul(targets_bb_mask[i])
        end
        local f = criterion:forward(outputs, {targets_class, targets_bb})
        cost_class_tmp = cost_class_tmp + criterion.criterions[1].output
        cost_box_tmp = cost_box_tmp + criterion.criterions[2].output

        -- update confusion
        for i=1,opt.batch_size do
            confusion:add(outputs[1][i]:squeeze(), targets_class[i])
        end

        xlua.progress(t, dataset:size())
    end

    confusion:updateValids()
    if datasetString == 'validation' then
        cost_class_tmp = cost_class_tmp/dataset:size()
        cost_box_tmp = cost_box_tmp/dataset:size()
        table.insert(cost.val.class, cost_class_tmp)
        table.insert(cost.val.box, cost_box_tmp)
        table.insert(error_valid, confusion.totalValid)
        print('<validation> validation accuracy: ' .. confusion.totalValid)
    elseif datasetString == 'test' then
        print('<testign> testset accuracy: ' .. confusion.totalValid)
    end
    confusion:zero()
end

-- Function to plot average cost from training set
-- (requires graphical display!)
function plotCost(avgWidth)
    if not gnuplot then
        require 'gnuplot'
    end
    local avgWidth = avgWidth or 50
    local costT = torch.Tensor(cost)
    local costX = torch.range(1, #cost)
    local nAvg = (#cost - #cost%avgWidth)/avgWidth
    local costAvg = torch.Tensor(nAvg)
    local costAvgX = torch.range(1, nAvg):mul(avgWidth)

    for i = 1,nAvg do
        costAvg[i] = costT[{{(i-1)*avgWidth+1, i*avgWidth}}]:mean()
    end
    --plots = {costT, costAvg}
    gnuplot.plot({'Mini batch cost',costX, costT},
                {'Mean over ' .. avgWidth .. ' batches', costAvgX, costAvg})
end

----- Higher level training and evaluation goes here -----

-- Example of training 12 epochs:
--for i=1,12 do
--    evaluateError('validation')
--    train()
--end

