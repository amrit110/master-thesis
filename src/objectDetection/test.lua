require 'image'
dofile 'nms.lua'
require 'paths'
require 'groupBoxes'
require 'qtwidget'

local dataPath = '/mnt/data/KITTI_Object_Detection/'
local imagesPath = '/mnt/data/cityscapes_processed/test/inputs/munich/'

torch.setdefaulttensortype('torch.FloatTensor')
function test()
    --require 'matio'
    model:evaluate()
    local inputs = loadTestBatch()
    collectgarbage()
    local out = model:forward(inputs:cuda())

    -- Apply softmax
    --local softnet = nn.Sequential():add(nn.SpatialSoftMax())
    --local pred = softnet:forward(out[1][1]:float())
    local pred = out[1][1]
    -- Plot image and heatmap
    local win = qtwidget.newwindow(inputs:size(4), inputs:size(3), 'BB plotting')
    local win1 = qtwidget.newwindow(inputs:size(4), inputs:size(3), 'Image')
    local heatmapActual = torch.zeros(inputs:size())
    local heatmapCar = image.scale(pred[3]:float(),'*4')
    local heatmapPed = image.scale(pred[4]:float(),'*4')   
    local heatmapCyc = image.scale(pred[5]:float(),'*4')   
    heatmapActual[{{},{2}}] = heatmapCar
    heatmapActual[{{},{1}}] = heatmapPed
    heatmapActual[{{},{3}}] = heatmapCyc
    inputs = inputs - torch.cmul(inputs,heatmapActual) + heatmapActual:mul(inputs:max())
    image.display{image = heatmapActual, zoom = 1, win=win}
    image.display{image = inputs,zoom = 1,win=win1}
    local t = win:image():toTensor(3)
    local t1 = win1:image():toTensor(3)
    image.save('heatmaps/result.png',t)
    image.save('heatmaps/input.png',t1)
    print(('Input mean and std: %.4f , %.4f'):format(inputs:mean(), inputs:std()))
    print(('Raw output mean and std: %.4f , %.4f'):format(out[1][1]:mean(), out[1][1]:std()))
    pred = pred:float()
    --matio.save('heatmaps/test.mat',pred)
    model:clearState()
    collectgarbage()
end


function loadTestBatch()
    local folderPath = dataPath .. 'testing' .. '/'
    local imagesPath = folderPath .. 'image_2'
    local indexImage = torch.random(1,#paths.dir(imagesPath)-2) - 1
    local input = image.load(imagesPath .. '/' .. string.format('%06d.png',indexImage))
    input = image.scale(input, 1024,'simple')
    --local input = image.load('heatmaps/test_gbg.png')
    --input = input[{{1,3}}]
    local t = require 'transforms' 
    local meanstd = {
        mean = { 0.485, 0.456, 0.406 },
        std = { 0.229, 0.224, 0.225 },
    }
    local transform = t.Compose{
        --t.Scale(1),
        t.ColorNormalize(meanstd),
        --t.CenterCrop(224),
    }
    input = transform(input) 
    --input = input:mul(255)
    local rows = input:size(2)
    local cols = input:size(3)
    -- SCALE IMAGE SO ITS DIVISABLE BY STRIDE:
    input = input[{{},{1,rows-rows%32},{1,cols-cols%32}}]:clone()
    input = input:view(1,3,input:size(2), input:size(3))
    return input
end


function deprocess(img)
    local perm = torch.LongTensor{3,2,1}
    img = img:index(1,perm)
    return img
end


function inferImage(img, threshold, NMSthreshold)
    local qtwidget = require('qtwidget')
 
    -- Parameters
    local nOutput = opt.nClasses
    local threshold = threshold or 0.8
    local NMSthreshold = NMSthreshold or 0.3

    -- Load image and run through network
    model:evaluate()
    local height = img:size(3)
    local width = img:size(4)
    collectgarbage()
    local out = model:forward(img:cuda())
    --local softnet = nn.Sequential():add(nn.SpatialSoftMax())
    --local pred = softnet:forward(out[1]:float())
    local pred = out[1]:float()
    local heatmapActual = torch.zeros(img:size())
    --image.display{image = pred[1][3],zoom = 3}
    local heatmap = image.scale(pred[1][3],'*4')
    local heatmapPed = image.scale(pred[1][4],'*4')   
    local heatmapCyc = image.scale(pred[1][5],'*4')   
    heatmapActual[{{},{2}}] = heatmap
    heatmapActual[{{},{1}}] = heatmapPed
    heatmapActual[{{},{3}}] = heatmapCyc
    img = img - torch.cmul(img,heatmapActual) + heatmapActual:mul(img:max())
    -- Matrixes for local -> global coordinates
    local xRange = torch.range(0,width-opt.stride,opt.stride)
    local offsetX = xRange:repeatTensor(height/opt.stride,1)
    local yRange = torch.range(0,height-opt.stride,opt.stride)
    local offsetY = yRange:repeatTensor(width/opt.stride,1):transpose(2,1)
    
    -- Transform to global coordinates
    local globalOut = out[2]:float():clone()
    local ranges = out[3]:float():clone()
    for i=1,3,2 do
        globalOut[{{},{i}}]:add(offsetX)
        globalOut[{{},{i+1}}]:add(offsetY)
    end
    -- Flatten output
    local boxes = globalOut:transpose(4,2,1,3):clone():view(height*width/(opt.stride^2),4)
    local rangeList = ranges:transpose(4,2,1,3):clone():view(height*width/(opt.stride^2),1)
    local list = pred:transpose(4,2,1,3):clone():view(height*width/(opt.stride^2),nOutput)
    -- Create window and display image
    local win = qtwidget.newwindow(img:size(4), img:size(3), 'BB plotting')
    win:gbegin()
    win:showpage()
    image.display{image=img, win=win}
    -- Loop through classes to be displayed
    for j = 3,opt.nClasses do 
        local newList = list[{{},{j}}]:clone()
        newList = newList:squeeze()
        --newList = newList:squeeze()
        -- Global theshold suppression:
        local indices = torch.linspace(1,newList:size(1), newList:size(1)):long()
        local selected = indices[newList:ge(threshold)]
        if selected:numel() > 0 then
            -- NON MAX suppression:
            local boxesNew = boxes:clone()
            boxesNew = boxes:index(1,selected)
            newList = newList:index(1,selected)
            local ind = nms(boxesNew, NMSthreshold, newList)
            --boxesNew = groupBoxes(boxesNew,1,0.14)
            boxesNew = boxesNew:index(1,ind)
            newList = newList:index(1,ind)
            if boxesNew:numel() > 0 then
                n_img = boxesNew:size(1)
            else
                n_img = 0
            end
            local rangeListNew = rangeList:clone()
            rangeListNew = rangeList:index(1,selected)
            rangeListNew = rangeListNew:index(1,ind)

            -- Loop through boxes and draw on window
            local n_img = newList:size(1)
            --print('Plotting ' .. n_img .. ' bounding boxes using threshold ' .. threshold ..
            --    ' and non-max suppression ' .. NMSthreshold)
            for i = 1,n_img do
                local coords = boxesNew[i]:squeeze():totable()
                local w = (coords[3]-coords[1])
                local h = (coords[4]-coords[2])
                local x = (coords[1])
                local y = (coords[2])
                if j == 3 then
                    win:setcolor(1,1,0)
                elseif j == 4 then
                    win:setcolor(1,0,1)
                elseif j == 5 then
                    win:setcolor(0,1,1)
                end
                win:setlinewidth(2)
                win:rectangle(x,y,w,h) 
                win:setfont(qt.QFont{italic=false,bold=true,size=15})
                win:moveto(coords[1], coords[2]-1)
                win:show(string.format('%02.2f m',rangeListNew[i]:squeeze()))
                win:stroke()
                local t = win:image():toTensor(3)
                --image.save('heatmaps/result1.png',t)
                win:gend()
            end
        else
            print('No boxes above threshold ' .. threshold .. '!')
        end
    end
    
    -- Finally, display activations and clear model state
    model:clearState()
end


function inference(threshold, NMSthreshold)
    local img = loadTestBatch()
    inferImage(img, threshold,NMSthreshold)
end

 
