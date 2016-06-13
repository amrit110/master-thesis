require 'paths'
--sequencePath = '/mnt/data/cityscapes/leftImg8bit/train_extra/nuremberg/'
sequencePath = '/mnt/data/demoVideo/stuttgart_00/'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
require 'qtwidget'



function testNet()
    local matio = require 'matio'
    model:evaluate()
    local testImg, label= loadMiniBatch(1, 'val')
    --local testImg, label = getBatch(1)
    local batchSize = 1
    --local inputs, targets = trainLoader:sample(batchSize)
    local softMaxNet = nn.Sequential()
    softMaxNet:add(nn.SpatialSoftMax())
    local out = model:forward(testImg:cuda()):squeeze()
    out = out:float()
    local out = softMaxNet:forward(out)
    --local _,indicesToDisplay = torch.max(out,1)
    --imageToDisplay[2] = indicesToDisplay:eq(3):mul(255)
    --imageToDisplay[3] = indicesToDisplay:eq(4):mul(255)
    --image.display{image = imageToDisplay, zoom = 0.25}--]]
    local perm = torch.LongTensor{3, 2, 1}
    local imageToDisplay = testImg:squeeze():index(1,perm)
    local _,ind = torch.max(out,1)
    local output = colourise(ind)
    --image.display{image = out, zoom = 0.1}
    local win = qtwidget.newwindow(imageToDisplay:size(3), 400, 'BB plotting')
    image.display{image = torch.add(imageToDisplay,output)[{{},{1,400}}], zoom = 1,win=win}
    local t = win:image():toTensor(3)
    image.save('heatmaps/argmax.png',t)
    image.display{image = out,zoom = 0.1}
end


function colourise(input)
    input = input:squeeze()
    local output = torch.zeros(3,input:size(1),input:size(2))
    for i = 1,input:size(1) do
        for j = 1,input:size(2) do
            if input[i][j] == 1 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,0}
            elseif input[i][j] == 2 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 3 then
                output[{{},{i},{j}}] = torch.Tensor{244,35,232}
            elseif input[i][j] == 4 then
                output[{{},{i},{j}}] = torch.Tensor{70,70,70}
            elseif input[i][j] == 5 then
                output[{{},{i},{j}}] = torch.Tensor{102,102,156}
            elseif input[i][j] == 6 then
                output[{{},{i},{j}}] = torch.Tensor{190,153,153}
            elseif input[i][j] == 7 then
                output[{{},{i},{j}}] = torch.Tensor{153,153,153}
            elseif input[i][j] == 8 then
                output[{{},{i},{j}}] = torch.Tensor{250,170,30}
            elseif input[i][j] == 9 then
                output[{{},{i},{j}}] = torch.Tensor{220,220,0}
            elseif input[i][j] == 10 then
                output[{{},{i},{j}}] = torch.Tensor{107,142,35}
            elseif input[i][j] == 11 then
                output[{{},{i},{j}}] = torch.Tensor{152,251,152}
            elseif input[i][j] == 12 then
                output[{{},{i},{j}}] = torch.Tensor{70,130,180}
            elseif input[i][j] == 13 then
                output[{{},{i},{j}}] = torch.Tensor{220,20,60}
            elseif input[i][j] == 14 then
                output[{{},{i},{j}}] = torch.Tensor{255,0,0}
            elseif input[i][j] == 15 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 16 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,70}
            elseif input[i][j] == 17 then
                output[{{},{i},{j}}] = torch.Tensor{0,60,100}
            elseif input[i][j] == 18 then
                output[{{},{i},{j}}] = torch.Tensor{0,80,100}
            elseif input[i][j] == 19 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,230}
            elseif input[i][j] == 20 then
                output[{{},{i},{j}}] = torch.Tensor{119,11,32}
            end
        end
    end
    return output
end


function colourise3(input)
    input = input:squeeze()
    local output = torch.zeros(3,input:size(1),input:size(2))
    for i = 1,input:size(1) do
        for j = 1,input:size(2) do
            if input[i][j] == 1 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,0}
            elseif input[i][j] == 2 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 3 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 4 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 5 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 6 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 7 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 8 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 9 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 10 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 11 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 12 then
                output[{{},{i},{j}}] = torch.Tensor{128,64,128}
            elseif input[i][j] == 13 then
                output[{{},{i},{j}}] = torch.Tensor{255,0,0}
            elseif input[i][j] == 14 then
                output[{{},{i},{j}}] = torch.Tensor{255,0,0}
            elseif input[i][j] == 15 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 16 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 17 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 18 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 19 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            elseif input[i][j] == 20 then
                output[{{},{i},{j}}] = torch.Tensor{0,0,142}
            end
        end
    end
    return output
end





function sequenceInference()
    torch.setdefaulttensortype('torch.FloatTensor')
    model = torch.load('/home/amrkri/Master_Thesis/savedModels/01/model.t7')
    model = model:cuda()
    model:evaluate()
    --local img = image.load(sequencePath .. string.format('nuremberg_000000_%06d_leftImg8bit.png',0))
    local img = image.load(sequencePath .. string.format('%06d.png',0))
    img = image.scale(img,1024,'simple')
    local rows = img:size(2)
    local cols = img:size(3)
    local input = img:view(1,img:size(1),img:size(2),img:size(3))
    local pred = model:forward(input:cuda())
    local win = qtwidget.newwindow(pred:size(4), 420, 'Heatmap plotting')
    for i = 1,#paths.dir(sequencePath)-2 do
        --local input = image.load(sequencePath .. string.format('nuremberg_000000_%06d_leftImg8bit.png',i-1))
        local input = image.load(sequencePath .. string.format('%06d.png',i-1))
        input = image.scale(input,1024,'simple')
        input:mul(255)
        local perm = torch.LongTensor{3, 2, 1}
        input = input:index(1, perm)
        input[{{1},{},{}}]:add(-123.68)
        input[{{2},{},{}}]:add(-116.779)
        input[{{3},{},{}}]:add(-103.939)
        input = input:view(1,input:size(1),input:size(2),input:size(3))
        local softMaxNet = nn.Sequential()
        softMaxNet:add(nn.SpatialSoftMax())
        local out = model:forward(input:cuda())
        local pred = softMaxNet:forward(out:float()):squeeze()
        local predSupp = torch.zeros(pred:size())
        for i = 1,pred:size(1) do
            predSupp[i] = pred[i]:gt(0.6)
        end
        local _,ind = torch.max(predSupp:squeeze(),1)
        local output = colourise(ind)
        local imageToDisplay = input:squeeze():index(1,perm)
        win:gbegin()
        image.display{image = torch.add(imageToDisplay,output)[{{},{1,420}}],win = win}
        saveImage(i,win, 'heatmaps/')
        win:gend()
    end
    model:clearState()
end

function saveSequence()
    for i = 1,#paths.dir(sequencePath)-2 do
        local input = image.load(sequencePath .. string.format('%06d.png',i-1))
        input = image.scale(input,512,'simple')
        saveImage(i, input, 'images/')
    end
end

function saveImage(indexFile, window, where)
    --local t = window
    local t = window:image():toTensor(3)
    image.save('resultVideos/'.. where .. string.format('%06d',indexFile) .. '.png', t)
end

