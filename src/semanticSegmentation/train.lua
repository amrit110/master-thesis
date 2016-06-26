--This is the main training script used for semantic segmentation. The script
--is used for both training and validation

--cost for monitoring plot
trainingCost = {}
validationCost = {}

-- Criterion to be used. Spatial training, so we use the CUDNN version of
-- spatial cross-entropy criterion
criterion_class = cudnn.SpatialCrossEntropyCriterion(opt.classWeights)
local criterion = criterion_class:cuda()

function plotCost(avgWidth)
    if not gnuplot then
        require 'gnuplot'
    end
    local avgWidth = avgWidth or 50
    local costT = torch.Tensor(trainingCost)
    local costX = torch.range(1, #trainingCost)
    local nAvg = (#cost - #cost%avgWidth)/avgWidth
    local costAvg = torch.Tensor(nAvg)
    local costAvgX = torch.range(1, nAvg):mul(avgWidth)

    for i = 1,nAvg do
        costAvg[i] = costT[{{(i-1)*avgWidth+1, i*avgWidth}}]:mean()
    end
    plots = {costT, costAvg}
    gnuplot.plot({'Mini batch cost',costX, costT},
                    {'Mean over ' .. avgWidth .. ' batches', costAvgX, costAvg})
end


function saveModel()
    model:clearState()
    torch.save('/mnt/data/pretrainedModels/networks/semanticSegmentation/semanticNetFull/model.t7',model)
end
--transfer learning. used to train deconvnet or full model--
function setParametersNet(mode)
    if mode == 'model' then
        parametersNetwork = nn.Sequential():add(model)
         print('Training the entire net now')
        parameters, gradParameters = parametersNetwork:getParameters()
    elseif mode == 'heads' then
        print('Training the upsampling net now')
        parametersNetwork = nn.Sequential():add(model:get(2))
        parameters, gradParameters = parametersNetwork:getParameters()
     end
end 

function train()
    model:training()
    local time = sys.clock()
    print('<trainer> on training set:')
    for t = 1,opt.nTrainingBatches*opt.batchSize,opt.batchSize do
        local inputs, targets = loadMiniBatch(opt.batchSize,'train')
        local inputs = inputs:cuda()
        targets = targets:cuda():squeeze()

        -- Evaluate cost and gradients
        local feval = function(x)
            collectgarbage()
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            table.insert(trainingCost,f)
            print('Cost of mini-batch: ' .. f)
            return f, gradParameters
        end

        -- Perform SGD
        optim.sgd(feval, parameters, sgdState)
        -- display progress
        xlua.progress(t, opt.nTrainingBatches*opt.batchSize)
    end

    print('<trainer> output cost of last batch: ' .. criterion.output)
    model:clearState()
    collectgarbage()
end

--can be used for validation. some errors with cudnn spatialfullconvolution.
--So, did not use for the thesis.
function validate()
    model:evaluate()
    local time = sys.clock()
    print('<trainer> on validation set:')
    for t = 1,opt.nValidationBatches*opt.batchSize,opt.batchSize do
        local inputs, targets = loadMiniBatch(opt.batchSize,'val')
        -- Evaluate cost and gradients
        local outputs = model:forward(inputs:cuda())
        targets = targets:cuda():squeeze()
        local f = criterion:forward(outputs, targets)
        table.insert(validationCost,f)
        print('Cost of mini-batch: ' .. f)
        -- display progress
        xlua.progress(t, opt.nValidationBatches*opt.batchSize)
    end

    print('<trainer> output cost of last batch: ' .. criterion.output)
    model:clearState()
    collectgarbage()
end




