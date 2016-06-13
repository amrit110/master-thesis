-- This file should be used ONLY by the donkeys
-- For testing or so, need other functions

require 'paths'
require 'image'
local dir = require 'pl.dir'
torch.setdefaulttensortype('torch.FloatTensor')

local dPath = '/mnt/data/KITTI/data/'
local dirs = dir.getdirectories(dPath)
--local imageSize = {375-23,1242-26}
--local maskSize = {11*8,38*8}
local setPaths = {}
local nImagesPerSet = {}

for k, dir in ipairs(dirs) do
    setPaths[k] = dPath .. paths.basename(dir) .. '/'
    local path = dPath .. paths.basename(dir) .. '/image_02/data/'
    nImagesPerSet[k] = (#paths.dir(path)) - 2
end

local function blockMatrix(alignment)
    local block = torch.Tensor(opt.maskSize[1],opt.maskSize[2])
    if alignment == 'x' then
        for i=0,37 do
            block[{{},{i*8+1,(i+1)*8}}]:fill(i*32)
        end
    elseif alignment == 'y' then
        for i=0,10 do
            block[{{i*8+1,(i+1)*8}}]:fill(i*32)
        end
    end
    return block:float()
end

local xOffset = blockMatrix('x')
local yOffset = blockMatrix('y')

function loadImageMaskAndBox(set,index,coordinateSystem)
    assert(index <= nImagesPerSet[set])
    -- Contruct path + file names and load image/mask pair
    local imagePath = tostring(setPaths[set] .. 'image_02/data/' .. string.format("%010u",index-1) .. '.png')
    local maskPath = setPaths[set] .. 'masks/' .. string.format("%010u",index-1) .. '.t7'
    local boxPath = setPaths[set] .. 'boxes/' .. string.format("%010u",index-1) .. '.t7'
    local img = image.load(imagePath,3,'float')
    img = preprocess(img)
    local mask = torch.load(maskPath):float() -- ??
    local box = torch.load(boxPath):float() -- ??

    -- Draw translation offset for the mask (0-5)
    local trX = torch.random(0,5)
    local trY = torch.random(0,5)
    -- Crop image and mask accordingly
    mask = image.crop(mask, trX, trY, opt.maskSize[2]+trX, opt.maskSize[1]+trY)
    box = image.crop(box, trX, trY, opt.maskSize[2]+trX, opt.maskSize[1]+trY)
    img = image.crop(img,trX*4,trY*4,opt.cropSize[2]+trX*4,opt.cropSize[1]+trY*4)
    -- Adjust box coordinates
    if coordinateSystem == 'local' then
        box[1]:add(-trX*4):add(-xOffset):cmul(mask)
        box[2]:add(-trY*4):add(-yOffset):cmul(mask)
        box[3]:add(-trX*4):add(-xOffset):cmul(mask)
        box[4]:add(-trY*4):add(-yOffset):cmul(mask)
    else
        box[1]:add(-trX*4):cmul(mask)
        box[2]:add(-trY*4):cmul(mask)
        box[3]:add(-trX*4):cmul(mask)
        box[4]:add(-trY*4):cmul(mask)
    end
    -- 0.5 chance to return horizontally flipped image
    --If math.random(1) < 0.5 then
    --    mask = image.hflip(mask)
    --    img = image.hflip(img)
    --end
    return img, mask, box
end

function drawSet()
    return torch.multinomial(torch.Tensor(nImagesPerSet), 1, true)
end

function getBatch(quantity,coordinateSystem)
    local images = torch.Tensor(quantity,3,opt.cropSize[1],opt.cropSize[2])
    local masks = torch.Tensor(quantity,opt.maskSize[1],opt.maskSize[2])
    local boxes = torch.Tensor(quantity,4,opt.maskSize[1],opt.maskSize[2])
    --local ids = {set={}, index={}}

    for i=1,quantity do
        --local class = torch.random(1, #classes)
        -- Draw uniformly among images: (not among sets)
        local set = drawSet():squeeze()
        local index = torch.random(1,nImagesPerSet[set])
        local img, mask, box = loadImageMaskAndBox(set,index,coordinateSystem)
        images[i] = img
        masks[i] = mask:add(1)
        boxes[i] = box
        --ids.set[i] = set
        --ids.index[i] = index
    end
    --images:add(-118.380948/255):div(61.896913/255)
    --masks:mul(-1):add(2)
    return images, {masks, boxes}
end


function preprocess(img)
    --local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    --local perm = torch.LongTensor{3, 2, 1}
    --img = img:index(1, perm):mul(256.0)
    --mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    --img:add(-1, mean_pixel)
    local perm = torch.LongTensor{3, 2, 1}
    img:mul(255)
    img = img:index(1, perm)
    img[1]:add(-123.68)
    img[2]:add(-116.779)
    img[3]:add(-103.939)
    --img = img:view(1,3,img:size(2),img:size(3))

    return img
end

function deprocess(img)
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
end




----------- DUMMY FUNCTIONS -------------

function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

-- Filler-in function to be able to implement training
-- prior to having available data
function getDummyBatch(quantity, coordinateSystem)

    -- Normalized images
    local images = torch.rand(quantity,3,opt.cropSize[1],opt.cropSize[2]):mul(255)
    images[{{},{1}}]:add(-100)
    images[{{},{2}}]:add(-100)
    images[{{},{3}}]:add(-100)
    
    -- Mask values of 0 or 1
    --local masks = torch.rand(quantity,opt.maskSize[1],opt.maskSize[2]):round():add(1)
    local masks = torch.zeros(quantity,opt.maskSize[1],opt.maskSize[2]):round():add(1)

    -- Box values between -80, -40 and 40, 80
    local boxes = torch.rand(quantity,4,opt.maskSize[1],opt.maskSize[2]):mul(40)
    boxes[{{},{1}}]:add(-80)
    boxes[{{},{2}}]:add(-80)
    boxes[{{},{3}}]:add(80)
    boxes[{{},{4}}]:add(80)

    return images, {masks, boxes}
end

