
local batchNumber = 0
local validationCost

-- High level training function, executing 'opt.nBatches' mini batches
function validate()
    batchNumber = 0
    model:evaluate()
    validationCost = {}
    print('Evaluating on validation set!')
    for i=1,opt.nValidationBatches do
        donkeys:addjob(
            function()
                local inputs, labels, outputMask = loadMiniBatch(opt.batchSize,'train','validation')
                return inputs, labels, outputMask
            end,
            validateBatch
        )
    end
    donkeys:synchronize()
    
    model:clearState()
    collectgarbage()

    -- Calculate avergae cost and store
    local s = 0
    for i, val in pairs(validationCost) do
        s = s + val
    end
    table.insert(overallCost.validation, s/#validationCost)
    print(('Validation average cost: %.4f'):format(s/#validationCost))
end

local timer = torch.Timer()
local dataTimer = torch.Timer()


-- Function for training a single mini batch
function validateBatch(inputsCPU, labelsCPU, outputMaskCPU)
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
    local objectMask = torch.ones(labels[1]:size())
    local dontCareMask = labels[1][{{},{2}}]:mul(-1):add(1) 
    ----
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
    --local gradOutputs = criterion:backward(outputs, labels)
    --model:backward(inputs, gradOutputs)
    --return err, gradParameters
    ----
    batchNumber = batchNumber + 1
    table.insert(validationCost,err)
    --print(('Minibatch: [%d/%d]\t Time %.4f Cost %.4f, DataLoadingTime %.3f'):format(
    --    batchNumber, opt.nBatches, timer:time().real, err, dataLoadingTime))

    dataTimer:reset()
    collectgarbage()
end




