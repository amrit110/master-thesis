--[[
  This file is run by each thread used for data loading.
  Containing files for loading original KITTI images, taking crops
  and creating propriate labels to be used in training.
--]]

require 'paths'
require 'image'
require 'nn'
local t = require 'transforms'
dir = require 'pl.dir'
torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
--file with function to create list of objects from the kitti dataset--
dofile 'kittiDataSetup.lua'
local allObjectsList, allImagePaths, allAnnotationPaths = makeObjectsList()
--parameters of masks and crop-masks used in training--
local shrinkFactor = 0.05
local maskWidth = torch.ceil(opt.imgSize[2]/opt.stride)
local maskHeight = torch.ceil(opt.imgSize[1]/opt.stride)
local cropMaskWidth = opt.cropSize[1]/opt.stride
local cropMaskHeight = opt.cropSize[1]/opt.stride

local dataPath = '/mnt/data/KITTI_Object_Detection/'
--mean and standard deviation used for resnet34--
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}


function file_exists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

function lines_from(file)
    if not file_exists(file) then return {} end
    local lines = {}
    for line in io.lines(file) do 
        lines[#lines + 1] = line
    end
    return lines
end


local transform = t.Compose{
    t.ColorNormalize(meanstd),
}

function plotBB()
    -- The function get the bounding box coordinates in the offset form and
    -- re-transfoms them to the original coordinates corresponding to the crop
    -- alone
    local indexMatrixX = torch.zeros(cropMaskHeight,cropMaskWidth)
    local indexMatrixY = torch.zeros(cropMaskHeight,cropMaskWidth)
    for i = 1,cropMaskHeight do
        indexMatrixX[{{i},{}}] = torch.range(0,opt.cropSize[1]-opt.stride,opt.stride)
    end
    for i = 1,cropMaskWidth do
        indexMatrixY[{{},{i}}] = torch.range(0,opt.cropSize[1]-opt.stride,opt.stride)
    end
    local img, label= loadMiniBatch(1, 'train')
    --local perm = torch.LongTensor{3, 2, 1}
    img = img:squeeze()
    label = label[2]:squeeze()
    --img = img:index(1, perm)
    local qtwidget = require('qtwidget')
    -- PLOTTING
    local win = qtwidget.newwindow(img:size(3), img:size(2), 'BB plotting')
    win:gbegin()
    win:showpage()
    image.display{image=img, win=win}
    --local list = torch.Tensor{{20,20,40,50},{100,200,150,210}
    local box = label:clone()
    box[{{1},{},{}}]  = box[{{1},{},{}}] + indexMatrixX
    box[{{2},{},{}}]  = box[{{2},{},{}}] + indexMatrixY
    box[{{3},{},{}}]  = box[{{3},{},{}}] + indexMatrixX
    box[{{4},{},{}}]  = box[{{4},{},{}}] + indexMatrixY
    box = box:view(1,4,cropMaskHeight,cropMaskWidth):transpose(4,2,1,3)
    list = box:clone():view(cropMaskWidth*cropMaskHeight,4)
    draw_bb(list, 1, win, 'red')
end

function draw_bb(list, win_zoom, win, color)
    local n_img = list:size(1)
    for i = 1,n_img do
        if list[i]:sum() ~= 0 then
            local coords = list[i]:squeeze():totable()
            local w = (coords[3]-coords[1])*win_zoom
            local h = (coords[4]-coords[2])*win_zoom
            local x = (coords[1])*win_zoom
            local y = (coords[2])*win_zoom
            win:setcolor(color)
            win:setlinewidth(2)
            win:rectangle(x,y,w,h)
            win:stroke()
            win:gend()
        end
    end
end


function plotMasks()
    local nMasks = 1
    img, label = loadMiniBatch(1,'train','test')
    img = img:squeeze()
    label = label[1][1]
    --local perm = torch.LongTensor{3, 2, 1}
    --img = img:index(1, perm) 
    --local mask1 = torch.add(torch.add(label:eq(3),label:eq(4)),label:eq(5))
    --mask1 = torch.add(torch.add(label[3],label[4]),label[5])
    mask1 = label[3]
    --mask1:gt(0):double()
    --local mask2 = label:eq(2)
    --mask2:gt(0):double()
    local mask2 = label[2]
    local newMask1 = torch.Tensor(nMasks,1,mask1:size(1)*4, mask1:size(2)*4)
    local newMask2 = torch.Tensor(nMasks,1,mask2:size(1)*4, mask2:size(2)*4)
    for i = 1,nMasks do
        newMask1[i] = image.scale(mask1, mask1:size(2)*4, mask1:size(1)*4,'simple')
        newMask2[i] = image.scale(mask2, mask2:size(2)*4, mask2:size(1)*4,'simple')
    end
    local maskActual = torch.zeros(img:size())
    maskActual[{{2},{},{}}] = newMask1
    maskActual[{{1},{},{}}] = newMask2
    img = img - torch.cmul(img,maskActual) + maskActual:mul(img:max())
    local qtwidget = require('qtwidget')
    -- PLOTTING
    local win = qtwidget.newwindow(img:size(3), img:size(2), 'Mask plotting')
    win:gbegin()
    win:showpage()
    image.display{image=img, win=win}
    local t = win:image():toTensor(3)
end

function giveMeAnObject()
    --pick random object. currently only for 3 classes but can be extended to
    --more. object oriented programming (classes) would be so much better here!
    local randomThrow = torch.uniform()
    local objectIndex, imagePath, object, objects
    if (randomThrow < (1/3)) then
        objectIndex = math.random(1,#allObjectsList.cars.image) 
        imagePath = allObjectsList.cars.image[objectIndex]
        object = allObjectsList.cars.details[objectIndex]
        objects = lines_from(allObjectsList.cars.annotations[objectIndex])
   elseif ((randomThrow >= (1/3)) and (randomThrow < (2/3))) then
        objectIndex = math.random(1,#allObjectsList.pedestrians.image) 
        imagePath = allObjectsList.pedestrians.image[objectIndex]
        object = allObjectsList.pedestrians.details[objectIndex]
        objects = lines_from(allObjectsList.pedestrians.annotations[objectIndex])
    elseif ((randomThrow >= (2/3)) and (randomThrow < (3/3))) then
        objectIndex = math.random(1,#allObjectsList.cyclists.image) 
        imagePath = allObjectsList.cyclists.image[objectIndex]
        object = allObjectsList.cyclists.details[objectIndex]
        objects = lines_from(allObjectsList.cyclists.annotations[objectIndex])
    end
    return imagePath, object, objects
end
 
function chooseWhatToMake()
    local randomTry = torch.uniform()
    local imagePath, object, objects
    if randomTry < 1 then
        repeat 
            imagePath, object, objects = giveMeAnObject()
            local words = {}
            for word in object:gmatch("%S+") do
                table.insert(words,word)
            end
        until (words[3] == '0')
        return imagePath, object, objects
    else
        local imageIndex = math.random(1,#allImagePaths)
        imagePath = allImagePaths[imageIndex]
        objects = lines_from(allAnnotationPaths[imageIndex]) 
        return imagePath, nil, objects
    end
end


function makeCropAndMask()
    local imagePath, object, objects = chooseWhatToMake()
    local input = image.load(imagePath)    
    local imgHeight = input:size(2)
    local imgWidth = input:size(3)
    local mask = torch.zeros(opt.nClasses,maskHeight,maskWidth)
    mask[1]:fill(1)
    local bbLabel = torch.zeros(5,maskHeight,maskWidth)
    local x1 = {}
    local x2 = {}
    local y1 = {}
    local y2 = {}
    local bbCoord = {}
    local bbCoords = {}
    local range = {}
    local objectType = {} -- 1 is for background, 2 is for don't train class (includes dontCare, Misc, Tram and Person-Sitting, 3 is for Car, 4 is for Truck, 5 is for Van, 6 for Pedestrian, 7 for Cyclist)
    local objectOcclusionState = {}
    for i = 1, #objects do
        local words = {}
        for word in objects[i]:gmatch("%S+") do
            table.insert(words,word)
        end
        bbCoord = {words[5]+1,words[6]+1,words[7]+1,words[8]+1}
        local centerx = bbCoord[1] + (bbCoord[3]-bbCoord[1])/2 
        local centery = bbCoord[2] + (bbCoord[4]-bbCoord[2])/2 
        local w = (bbCoord[3] - bbCoord[1]) *math.sqrt(shrinkFactor)
        local h = (bbCoord[4] - bbCoord[2]) *math.sqrt(shrinkFactor)
        centerx = math.floor((centerx)/opt.stride)
        centery = math.floor((centery)/opt.stride)
        h = h/opt.stride
        w = w/opt.stride
        table.insert(x1,math.max(1, math.min(maskWidth, (centerx - w/2))))
        table.insert(x2,math.max(1, math.min(maskWidth, (centerx + w/2))))
        table.insert(y1,math.max(1, math.min(maskHeight, (centery - h/2))))
        table.insert(y2,math.max(1, math.min(maskHeight, (centery + h/2))))
        table.insert(bbCoords,bbCoord)
        table.insert(range,(words[12]^2+words[13]^2+words[14]^2)^(1/2))
        if ((words[1] == 'DontCare') or (words[2] == '2') or (words[3] == '3') or (words[1] == 'Person_sitting') or (words[1] == 'Van')) then
            table.insert(objectType,2)
        elseif words[1] == 'Car' then
            table.insert(objectType,3)
        elseif words[1] == 'Pedestrian' then
            table.insert(objectType,4)
        elseif words[1] == 'Cyclist' then
            table.insert(objectType,5)
        elseif ((words[1] == 'Misc') or (words[1] == 'Tram') or (words[1] == 'Truck')) then 
            table.insert(objectType,1)
        else 
            table.insert(objectType,1)
        end
        table.insert(objectOcclusionState,words[3])
    end
    local rangeSorted,indices = torch.sort(torch.Tensor(range),1,true)
    local objectsS = {}
    local x1S = {}
    local y1S = {}
    local x2S = {}
    local y2S = {}
    local bbCoordsS = {}
    local objectTypeS = {}
    local objectOcclusionStateS = {}
    local nObjects = #objects
    -- Sorting through objects according to the distance to them so that
    -- farther objects are included in the mask first and the rest are painted
    -- over them
    for i = 1,nObjects do
        table.insert(x1S,x1[indices[i]])
        table.insert(y1S,y1[indices[i]])
        table.insert(x2S,x2[indices[i]])
        table.insert(y2S,y2[indices[i]])
        table.insert(bbCoordsS,bbCoords[indices[i]])
        table.insert(objectTypeS,objectType[indices[i]])
        table.insert(objectOcclusionStateS,objectOcclusionState[indices[i]])
        table.insert(objectsS,objects[indices[i]])
    end
    for i = 1,nObjects do    
        if ((objectTypeS[i] ~=2) and (objectTypeS[i] ~= 1)) then
            if ((y1S[i]-1) > 0) and ((y2S[i]+1) < maskHeight+1) and ((x1S[i]-1) > 0) and ((x2S[i]+1) < maskWidth+1) then
                mask[{{2},{y1S[i]-1,y2S[i]+1},{x1S[i]-1,x2S[i]+1}}]:fill(1)
                mask[{{2},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(0)
            end
        end
        if (objectTypeS[i] == 2) then
            mask[{{2},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(1)
            mask[{{1},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(0)
        elseif (objectTypeS[i] == 3) then
            mask[{{3},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(1)
            mask[{{1},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(0)
        elseif (objectTypeS[i] == 4) then
            mask[{{4},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(1)
            mask[{{1},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(0)
        elseif (objectTypeS[i] == 5) then
            mask[{{5},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(1)
            mask[{{1},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(0)
        end
        bbLabel[{{1},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(bbCoordsS[i][1])
        bbLabel[{{2},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(bbCoordsS[i][2])
        bbLabel[{{3},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(bbCoordsS[i][3])
        bbLabel[{{4},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(bbCoordsS[i][4])
        bbLabel[{{5},{y1S[i],y2S[i]},{x1S[i],x2S[i]}}]:fill(rangeSorted[i]) 
    end
    local maskToReturn = torch.cat(mask,bbLabel,1)
    local x,y,w,h,centerx,centery,bbCoord, bkgFlag = {}
    if object == nil then
        print('background')
        centerx = torch.random(opt.cropSize[1]/2 , imgWidth - opt.cropSize[1]/2)
        centery = torch.random(opt.cropSize[1]/2, imgHeight - opt.cropSize[1]/2)
        x = centerx - opt.cropSize[1]/2
        y = centery - opt.cropSize[1]/2
        w = x + opt.cropSize[1] - 1
        h = y + opt.cropSize[1] - 1
    else
        print('object')
        local words = {}
        for word in object:gmatch("%S+") do
            table.insert(words,word)
        end
        bbCoord = {words[5]+1,words[6]+1,words[7]+1,words[8]+1}
        centerx = bbCoord[1] + (bbCoord[3]-bbCoord[1])/2 
        centerx = centerx + math.random(math.min(-opt.offsetCrop,-bbCoord[1]),math.min(opt.offsetCrop,imgWidth-bbCoord[3]))
        centery = bbCoord[2] + (bbCoord[4]-bbCoord[2])/2 
        centery = centery + math.random(math.min(-opt.offsetCrop,-bbCoord[2]),math.min(opt.offsetCrop,imgHeight-bbCoord[4]))
        x = centerx - math.ceil(opt.cropSize[1]/2)
        y = centery - math.ceil(opt.cropSize[1]/2) 
        w = x + opt.cropSize[1] - 1
        h = y + opt.cropSize[1] - 1
    end
    if (x < 1) or (y < 1) or (h > imgHeight) or (w > imgWidth) then
        print('exceeding dimensions')
        return nil, nil, 0
    end
    local crop = input[{{},{y,h},{x,w}}]
    local maskCenterx = centerx/opt.stride
    local maskCentery = centery/opt.stride
    local maskx1 = math.max(1, math.min(maskWidth, maskCenterx - math.floor(opt.cropSize[1]/(2*opt.stride))))
    local maskx2 = math.max(1, math.min(maskWidth, maskCenterx + math.floor(opt.cropSize[1]/(2*opt.stride))))
    local masky1 = math.max(1, math.min(maskHeight,maskCentery - math.floor(opt.cropSize[1]/(2*opt.stride))))
    local masky2 = math.max(1, math.min(maskHeight,maskCentery + math.floor(opt.cropSize[1]/(2*opt.stride))))
    if (masky2-masky1 ~= cropMaskHeight) or (maskx2-maskx1 ~= cropMaskWidth) then
        print('mask dimensions not met')
        return nil,nil,0
    end
    maskToReturn = maskToReturn[{{},{masky1,masky2-1},{maskx1,maskx2-1}}]
    maskToReturn[{{opt.nClasses+1},{},{}}]:add(-x)
    maskToReturn[{{opt.nClasses+2},{},{}}]:add(-y)
    maskToReturn[{{opt.nClasses+3},{},{}}]:add(-x)
    maskToReturn[{{opt.nClasses+4},{},{}}]:add(-y)
    for i = 1,cropMaskHeight-1 do
        for j = 1,cropMaskWidth-1 do
            maskToReturn[{{opt.nClasses+1},{i},{j}}]:add(-(j-1)*opt.stride) --The offsetting and storing bounding box coordinates according to the DenseBox paper
            maskToReturn[{{opt.nClasses+2},{i},{j}}]:add(-(i-1)*opt.stride)
            maskToReturn[{{opt.nClasses+3},{i},{j}}]:add(-(j-1)*opt.stride)
            maskToReturn[{{opt.nClasses+4},{i},{j}}]:add(-(i-1)*opt.stride)
        end
    end
    local objectMask = torch.add(maskToReturn[1],maskToReturn[2])
    objectMask = objectMask:gt(0)
    objectMask = objectMask:type('torch.FloatTensor')
    objectMask:mul(-1):add(1)
    for i = opt.nClasses+1,opt.nClasses+5 do
        maskToReturn[i] = torch.cmul(maskToReturn[i],objectMask)    
    end
    return crop, maskToReturn, objectMask, 1
end



function loadImageAndMask(folderPath, mode)
    local input, mask, validImage
    validImage = 0
    repeat
        input, mask, objectMask, validImage = makeCropAndMask()
    until validImage == 1
    input = transform(input)
    return input, mask, objectMask
end


function loadMiniBatch(batchSize, mode, submode)
    local inputs, labels, outputMasks, bbLabels
    local input, label
    local folderPath
    if mode == 'train' then
        inputs = torch.Tensor(batchSize, 3, opt.cropSize[1], opt.cropSize[1])
        labels = torch.Tensor(batchSize, opt.nClasses+5, cropMaskHeight, cropMaskWidth)
        outputMasks = torch.Tensor(batchSize,1, cropMaskHeight, cropMaskWidth)
        folderPath = dataPath .. 'training' .. '/'
    elseif mode == 'test' then
        folderPath = dataPath .. 'testing' .. '/'
    else
        print('Invalid Mode') 
    end
    if mode == 'train' then
        for i = 1, batchSize do
            input, label,outputMask = loadImageAndMask(folderPath, submode)
            inputs[i] = input
            labels[i] = label
            outputMasks[i] = outputMask
        end
    elseif mode == 'test' then
        local imagesPath = folderPath .. 'image_2'
        local indexImage = torch.random(1,#paths.dir(imagesPath)-2) - 1
        local input = image.load(imagesPath .. '/' .. string.format('%06d.png',indexImage))
        --[[local perm = torch.LongTensor{3,2,1}
        input = input:index(1,perm):mul(255)
        input[1]:add(-123.68)
        input[2]:add(-116.779)
        input[3]:add(-103.939)--]]
        input = transform(input)
        local rows = input:size(2)
        local cols = input:size(3)
        input = input[{{},{1,rows-rows%8},{1,cols-cols%8}}]:clone()
        input = input:view(1,3,input:size(2), input:size(3))
        return input,nil, nil
    end
    return inputs, {labels[{{},{1,opt.nClasses}}],labels[{{},{opt.nClasses+1,opt.nClasses+4}}],labels[{{},{opt.nClasses+5}}]}, outputMasks
end

function getClassWeights()
    -- Function used for calculating an estimate of the occurance of different
    -- classes in the dataset.
    local times = 1
    local results = torch.zeros(times,opt.nClasses)
    for i = 1,times do
        imgs, labs = loadMiniBatch(500,'train')
        hist = torch.histc(labs[1],opt.nClasses)
        hist = hist:div(torch.max(hist))
        local tmp = torch.ones(opt.nClasses)
        results[i] = torch.cdiv(tmp,hist)
    end
    finalResults = torch.mean(results,1)
    return finalResults
end




