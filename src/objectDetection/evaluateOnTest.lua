opt = {
    nClasses = 5,
    manualSeed = 10,
    stride = 4,
}
torch.setdefaulttensortype('torch.FloatTensor')
require 'paths'
require 'image'
dofile 'nms.lua'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'Tiling'
local qtwidget = require('qtwidget')
local testPath = '/mnt/data/KITTI_Object_Detection/training/images/'
local t = require 'transforms'
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
t.ColorNormalize(meanstd),
}
local nOutput = opt.nClasses
model = torch.load('networks/2605/model.t7'):cuda()
model:evaluate()

function createZeroPaddingNet(padRow,padCol)
    local zeroPaddingNet = nn.Sequential():add(nn.SpatialZeroPadding(0,padCol,0,padRow))
    return zeroPaddingNet
end


function inferSequence() 
local input = image.load(testPath .. string.format('%06d.png',0))
local rows = input:size(2)
local cols = input:size(3)
--input = input[{{},{1,rows-rows%32},{1,cols-cols%32}}]:clone()
input = input:view(1,3,input:size(2), input:size(3))
local win = qtwidget.newwindow(input:size(4), input:size(3), 'BB plotting')
    for i = 6981,#paths.dir(testPath)-2 do
        local input = image.load(testPath .. string.format('%06d.png',i-1))
        input = transform(input)
        local rows = input:size(2)
        local cols = input:size(3)        
        -- SCALE IMAGE SO ITS DIVISABLE BY STRIDE:
        local originalInput = input:clone()
        
        local paddingNet = createZeroPaddingNet(32-rows%32,32-cols%32)
        input = paddingNet:forward(input)
        --input = input[{{},{1,rows-rows%32},{1,cols-cols%32}}]:clone()
        input = input:view(1,3,input:size(2), input:size(3))
        local height = input:size(3)
        local width = input:size(4)
        local threshold = threshold or 0.8
        local NMSthreshold = NMSthreshold or 0.3
        collectgarbage()
        local out = model:forward(input:cuda())
        local pred = out[1]:float()
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
        local list = pred:transpose(4,2,1,3):clone():view(height*width/opt.stride^2,nOutput)
        win:gbegin()
        image.display{image=originalInput, win=win}
            -- SUPPRESSION GOES HERE:
        -- Global theshold suppression:
        local results = {}
        for l = 3, opt.nClasses do
            local newList = list[{{},{l}}]:clone()
            newList = newList:squeeze()
            local indices = torch.linspace(1,newList:size(1), newList:size(1)):long()
            local selected = indices[newList:ge(threshold)]
            if selected:numel() > 0 then
                -- NON MAX suppression:
                local boxesNew = boxes:clone()
                boxesNew = boxesNew:index(1,selected)
                newList = newList:index(1,selected)
                local ind = nms(boxesNew, NMSthreshold, newList)
                boxesNew = boxesNew:index(1,ind)
                newList = newList:index(1,ind)
                local rangeListNew = rangeList:clone()
                rangeListNew = rangeList:index(1,selected)
                rangeListNew = rangeListNew:index(1,ind)
                -- Loop through boxes and draw on window
                local n_img = newList:size(1)
                for k = 1,n_img do
                    -- Only plot boxes for cars:
                    local coords = boxesNew[k]:squeeze():totable()
                    coords[1] = ((coords[1] < 1) and 0) or coords[1]
                    coords[2] = ((coords[2] < 1) and 0) or coords[2]
                    coords[3] = ((coords[3] > originalInput:size(3)) and originalInput:size(3)) or coords[3]
                    coords[4] = ((coords[4] > originalInput:size(2)) and originalInput:size(3)) or coords[4]
                    if l == 3 then
                        local result = {'Car',-1,-1,-10,string.format('%.2f',coords[1]),string.format('%.2f',coords[2]),string.format('%.2f',coords[3]),string.format('%.2f',coords[4]),rangeListNew[k],-1,-1,-1000,-1000,-1000,-10,string.format('%.2f',newList[k])}
                        table.insert(results,result)
                        win:setcolor(1,0,1)
                    elseif l == 4 then
                        local result = {'Pedestrian',-1,-1,-10,string.format('%.2f',coords[1]),string.format('%.2f',coords[2]),string.format('%.2f',coords[3]),string.format('%.2f',coords[4]),rangeListNew[k],-1,-1,-1000,-1000,-1000,-10,string.format('%.2f',newList[k])}
                        table.insert(results,result)
                        win:setcolor(1,1,0)
                    elseif l == 5 then
                        local result = {'Cyclist',-1,-1,-10,string.format('%.2f',coords[1]),string.format('%.2f',coords[2]),string.format('%.2f',coords[3]),string.format('%.2f',coords[4]),rangeListNew[k],-1,-1,-1000,-1000,-1000,-10,string.format('%.2f',newList[k])}
                        table.insert(results,result)
                        win:setcolor(0,1,1)
                    end
                    local w = (coords[3]-coords[1])
                    local h = (coords[4]-coords[2])
                    local x = (coords[1])
                    local y = (coords[2])
                    win:setlinewidth(2)
                    win:rectangle(x,y,w,h) 
                    win:setfont(qt.QFont{italic=false,size=10})
                    win:moveto(coords[1], coords[2]-1)
                    win:stroke()
                end
                win:gend() 
            else
                win:gend()
            end
            writeTextFile(i-1,results)
            xlua.progress(i,#paths.dir(testPath)-2)
        end
    end
    model:clearState()
end

function writeTextFile(index,results)
    file = io.open(string.format('testResults/%06d.txt',index),"w")
    for i = 1,#results do
        for j = 1,#results[i] do
            file:write(tostring(results[i][j])," ")
        end
        file:write("\n")
    end
    file:close()
end



