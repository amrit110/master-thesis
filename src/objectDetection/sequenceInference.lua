--This script is sued for inference on sequence of images, to create a video
---Packages and other script dependencies--- 
require 'nn'
require 'cudnn'
require 'cunn'
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'Tiling'
require 'paths'
local dir  = require 'pl.dir'
require 'image'
dofile 'nms.lua'
--require 'groupBoxes'
local qtwidget = require('qtwidget')
local t = require 'transforms'
-------------------
---Normalisation--- 
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
t.ColorNormalize(meanstd),
}
-------------------
---Parameters------
opt = {
    nClasses = 5,
    cropSize = {224,224},
    maskSize = {56,56},
    stride = 4,
}

torch.setdefaulttensortype('torch.FloatTensor')



--Paths for different sequences of images

--sequencePath = '/mnt/data/KITTI_Object_Detection/testing/image_2/'
sequencePath = '/mnt/data/KITTI/data/2011_09_26_drive_0023_sync/'
--sequencePath = '/mnt/data/cityscapes/leftImg8bit/train_extra/nuremberg/'
--sequencePath = '/mnt/data/cityscapes_processed/demoVideo/stuttgart_00/'
sequencePath = sequencePath .. 'image_02/data/'
--------------------
---Zero Padding to ensure the proper size to network
function createZeroPaddingNet(padRow,padCol)
    local zeroPaddingNet = nn.Sequential():add(nn.SpatialZeroPadding(0,padCol,0,padRow))
    return zeroPaddingNet
end
--------------------

function loadModel(modelFolder)
    local model = torch.load('/mnt/data/pretrainedModels/networks/' .. modelFolder .. '/model.t7')
    collectgarbage()
    return model
end
--------------------
-- Fucntion for sorting a lua table
model = loadModel('autolivNet'):cuda()
model:evaluate()

function inferSequence() 
    local input = image.load(sequencePath .. string.format('%010d.png',1))
    --local input = image.load(sequencePath .. string.format('nuremberg_000000_%06d_leftImg8bit.png',0))
    --input = image.scale(input,1024,'simple')
    --input = input[{{},{1,420},{}}]:clone()
    local rows = input:size(2)
    local cols = input:size(3)
    input = input:view(1,3,input:size(2), input:size(3))
    local win = qtwidget.newwindow(input:size(4), input:size(3), 'BB plotting')
    for i = 1,#paths.dir(sequencePath)-2 do
        local input = image.load(sequencePath .. string.format('%010d.png',i))
        --local input = image.load(sequencePath .. string.format('nuremberg_000000_%06d_leftImg8bit.png',i-1))
        --input = image.scale(input,1024,'simple')
        --input = input[{{},{1,420},{}}]
        input = transform(input)
        local rows = input:size(2)
        local cols = input:size(3)
        -- SCALE IMAGE SO ITS DIVISABLE BY STRIDE:
        local originalInput = input:clone()        
        local paddingNet = createZeroPaddingNet(32-rows%32,32-cols%32)
        input = paddingNet:forward(input)
        input = input:view(1,3,input:size(2), input:size(3))
        local height = input:size(3)
        local width = input:size(4)
        local threshold = threshold or 0.8
        local NMSthreshold = NMSthreshold or 0.3
        collectgarbage()
        local out = model:forward(input:cuda())
        --local pred = softnet:forward(out[1]:float())
        local pred = out[1]:float()
        local heatmapActual = torch.zeros(input:size())
        local heatmapCar = image.scale(pred[1][3]:float(),'*4')
        local heatmapPed = image.scale(pred[1][4]:float(),'*4')   
        local heatmapCyc = image.scale(pred[1][5]:float(),'*4')   
        heatmapActual[{{},{2}}] = heatmapCar
        heatmapActual[{{},{1}}] = heatmapPed
        heatmapActual[{{},{3}}] = heatmapCyc
        input = input - torch.cmul(input,heatmapActual) + heatmapActual:mul(input:max())
        local xRange = torch.range(0,width-opt.stride,opt.stride)
        local offsetX = xRange:repeatTensor(height/opt.stride,1)
        local yRange = torch.range(0,height-opt.stride,opt.stride)
        local offsetY = yRange:repeatTensor(width/opt.stride,1):transpose(2,1)
        local globalOut = out[2]:float():clone()
        local ranges = out[3]:float():clone()
        for j=1,3,2 do
            globalOut[{{},{j}}]:add(offsetX)
            globalOut[{{},{j+1}}]:add(offsetY)
        end
        local boxes = globalOut:transpose(4,2,1,3):clone():view(height*width/opt.stride^2,4)
        local rangeList = ranges:transpose(4,2,1,3):clone():view(height*width/opt.stride^2,1)
        local list = pred:transpose(4,2,1,3):clone():view(height*width/opt.stride^2,opt.nClasses)
        win:gbegin()
        image.display{image=input, win=win}
        for k = 3,opt.nClasses do
            local newList = list[{{},{k}}]:clone()
            newList = newList:squeeze()
            local indices = torch.linspace(1,newList:size(1), newList:size(1)):long()
            local selected = indices[newList:ge(threshold)]
            if selected:numel() > 0 then
                local n_img
                -- NON MAX suppression:
                local boxesNew = boxes:clone()
                boxesNew = boxesNew:index(1,selected)
                newList = newList:index(1,selected)
                --boxesNew = groupBoxes(boxesNew,3,0.15)
                if boxesNew:numel() > 0 then
                    n_img = boxesNew:size(1)
                else
                    n_img = 0
                end
                local ind = nms(boxesNew, NMSthreshold, newList)
                boxesNew = boxesNew:index(1,ind)
                newList = newList:index(1,ind)
                local rangeListNew = rangeList:clone()
                local rangeListNew = rangeListNew:index(1,selected)
                rangeListNew = rangeListNew:index(1,ind)
                -- Loop through boxes and draw on window
                local n_img = boxesNew:size(1)
                for i = 1,n_img do
                -- Only plot boxes for cars:
                    local coords = boxesNew[i]:squeeze():totable()
                    local w = (coords[3]-coords[1])
                    local h = (coords[4]-coords[2])
                    local x = (coords[1])
                    local y = (coords[2])
                    if (k == 3) then
                        win:setcolor(1,1,0)
                    elseif (k == 4) then
                        win:setcolor(1,0,0)
                    elseif (k == 5) then
                        win:setcolor(0,1,1)
                    end
                    win:setlinewidth(2)
                    win:rectangle(x,y,w,h) 
                    win:setfont(qt.QFont{italic=false,bold=true,size=10})
                    win:moveto(coords[1], coords[2]-1)
                    win:show(string.format('%02.2fm',rangeListNew[i]:squeeze()))
                    win:stroke()
                end
                saveImage(i,win, 'images/')
                win:gend()
            else
                saveImage(i,win, 'images/')
                win:gend()
            end
        end
    end
    model:clearState()
end

function saveImage(indexFile, window, where)
    local t = window:image():toTensor(3)
    image.save('results/'.. where .. string.format('%06d',indexFile) .. '.png', t)
end
