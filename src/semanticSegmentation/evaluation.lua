--script used to generate the png files for evaluation on the cityscapes.

require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'image'
local dir = require 'pl.dir'
torch.setdefaulttensortype('torch.FloatTensor')

function mysplit(inputstr,sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i+1
    end
    return t
end

local dataPath = '/mnt/data/cityscapes/'
--specify the path to save the results for submitting to server
local resultsDir = 'results/'
local model = torch.load('/mnt/data/pretrainedModels/networks/semanticSegmentation/semanticNetFull/model.t7')
model:evaluate()

--rescaling image and normalising to be used for input to the network
function processImage(img)
    local rescaledHeight = img:size(2)/2
    local rescaledWidth = img:size(3)/2
    img:mul(255)
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    img[{{1},{},{}}]:add(-123.68)
    img[{{2},{},{}}]:add(-116.779)
    img[{{3},{},{}}]:add(-103.939)
    local resizedImg = image.scale(img,'*0.5','simple')
    resizedImg = resizedImg:view(1,3,resizedImg:size(2),resizedImg:size(3))
    return resizedImg
end

--takes the output from the network and sets the original labels according to
--the ciyscapes definition
function resetLabels(indices)
    local result = indices:clone()
    result[indices:eq(1)] = 0
    result[indices:eq(2)] = 7
    result[indices:eq(3)] = 8
    result[indices:eq(4)] = 11
    result[indices:eq(5)] = 12
    result[indices:eq(6)] = 13
    result[indices:eq(7)] = 17
    result[indices:eq(8)] = 19
    result[indices:eq(9)] = 20
    result[indices:eq(10)] = 21
    result[indices:eq(11)] = 22
    result[indices:eq(12)] = 23
    result[indices:eq(13)] = 24
    result[indices:eq(14)] = 25
    result[indices:eq(15)] = 26
    result[indices:eq(16)] = 27
    result[indices:eq(17)] = 28
    result[indices:eq(18)] = 31
    result[indices:eq(19)] = 32
    result[indices:eq(20)] = 33
    return result
end


function validate()
    local imagesPath = dataPath .. 'leftImg8bit/' .. 'test/'
    local filePaths = dir.getallfiles(imagesPath)
    for i, filePath in pairs(filePaths) do
        local input = image.load(filePath)
        input = processImage(input)
        local out = model:forward(input:cuda()):squeeze():float() 
        local resizedOut = image.scale(out,'*2','bilinear')
        local _,ind = torch.max(resizedOut,1)
        ind = ind:float()
        local result = resetLabels(ind):byte()
        local namesplits = mysplit(mysplit(filePath,'/')[7],'_')
        local saveName = namesplits[1] .. '_' .. namesplits[2] .. '_' .. namesplits[3] .. '.png'
        image.save(resultsDir .. saveName, result)
        xlua.progress(i, #filePaths)
    end
end
        
