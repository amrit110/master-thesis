require 'optim'


cost = cost or {}
local batchNumber
local trainingCost

-- High level training function, executing 'opt.nBatches' mini batches
function train()
    trainingCost = {}
    batchNumber = 0
    model:training()
    for i=1,opt.nBatches do
        donkeys:addjob(
            function()
                local inputs, labels, outputMask = loadMiniBatch(opt.batchSize,'train','train')
                return inputs, labels, outputMask
            end,
            trainBatch
        )
    end
    donkeys:synchronize()
    
    model:clearState()
    collectgarbage()
    
    -- Calculate avergae cost and store
    local s = 0
    for i, val in pairs(trainingCost) do
        s = s + val
    end
    table.insert(overallCost.training, s/#trainingCost)
end

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- Defining the parameters to train.
--local parametersNetwork = nn.Sequential():add(model)
-- Defining the parameters to train (DEFAULT: HEADS)
local parametersNetwork = nn.Sequential():add(model:get(2))
local parameters, gradParameters = parametersNetwork:getParameters()

-- Use to switch between training head and model
function setParameterNetwork(mode)
    if mode == 'model' then
        parametersNetwork = nn.Sequential():add(model)
        parameters, gradParameters = parametersNetwork:getParameters()
    elseif mode == 'heads' then
        parametersNetwork = nn.Sequential():add(model:get(2))
        parameters, gradParameters = parametersNetwork:getParameters()
    end
end

-- Just to confirm what parameters are updated in training
function printParameterSize()
    print(parameters:size())
end



-- Function for training a single mini batch
function trainBatch(inputsCPU, labelsCPU, outputMaskCPU)
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    local labels = labelsCPU
    local inputs = inputsCPU:cuda()
    local outputMask = outputMaskCPU:cuda()
    labels[1] = labels[1]:clone():cuda()
    labels[2] = labels[2]:clone():cuda()
    labels[3] = labels[3]:clone():cuda()
    local trainMask
    local err, outputs
    --local objectMask = torch.ones(labels[1]:size())
    local dontCareMask = labels[1][{{},{2}}]:mul(-1):add(1) 
    feval = function(x)
        model:zeroGradParameters()
        outputs = model:forward(inputs)
       for i = 1,4 do
            outputs[2][{{},{i}}] = torch.cmul(outputs[2][{{},{i}}],outputMask)
        end
        for i = 1,opt.nClasses do
            outputs[1][{{},{i}}] = torch.cmul(outputs[1][{{},{i}}],dontCareMask)
        end
        labels[1][{{},{2}}]:zero()
        outputs[3] = torch.cmul(outputs[3],outputMask)
        err = criterion:forward(outputs, labels)
        local gradOutputs = criterion:backward(outputs, labels)
        model:backward(inputs, gradOutputs)
        return err, gradParameters
    end
    optim.sgd(feval, parameters, opt.sgdState)
    batchNumber = batchNumber + 1
    table.insert(cost,err)
    table.insert(trainingCost, err)
    print(('Minibatch: [%d/%d]\t Time %.4f Cost %.4f, DataLoadingTime %.3f'):format(
        batchNumber, opt.nBatches, timer:time().real, err, dataLoadingTime))

    dataTimer:reset()
    collectgarbage()
end

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



