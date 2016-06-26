--This file has the function to load a minibatch from the cityscapes dataset.
--The dataset is expected to be arranged into training, validation and test
--data. The inputs and labels are all PNG files and loaded as torch tensors and
--returned as a torch tensor for training.


require 'paths'
require 'image'
math.randomseed(os.time())
dir = require 'pl.dir'
local t = require 'transforms'

local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}

function file_exists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end
--extract lines from a .txt file--
function lines_from(file)
    if not file_exists(file) then return {} end
    local lines = {}
    for line in io.lines(file) do
        lines[#lines + 1] = line
    end
    return lines
end
--used for splitting strings--
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

local transform = t.Compose{
    --t.Scale(1),
    t.ColorNormalize(meanstd),
    --t.CenterCrop(224),
    }
--Global Variables--
inputWidth = 2048
inputHeight = 1024
reScaledWidth = 1024
reScaledHeight = 512
flipProbability = 0.5
dataPath = '/mnt/data/cityscapes_processed/'


function loadInputAndLabel(folderPath, index)
    local inputsPath = folderPath .. 'inputs'
    local labelsPath = folderPath .. 'labels'
    local directoriesInputs = dir.getdirectories(inputsPath)
    local directoriesLabels = dir.getdirectories(labelsPath)
    indexFolder = math.random(1,#directoriesInputs)
    indexImage = math.random(1,#paths.dir(directoriesInputs[indexFolder])-2)
    local input = image.load(directoriesInputs[indexFolder] .. '/' .. string.format('%06d.png',indexImage))
    local label = image.load(directoriesLabels[indexFolder] .. '/' .. string.format('%06d.png',indexImage))
    label = torch.floor(label:mul(255))
    input:mul(255)
    local perm = torch.LongTensor{3, 2, 1}
    input = input:index(1, perm)
    input[{{1},{},{}}]:add(-123.68)
    input[{{2},{},{}}]:add(-116.779)
    input[{{3},{},{}}]:add(-103.939)
    input = image.scale(input,reScaledWidth,reScaledHeight,'simple')
    label = image.scale(label,reScaledWidth,reScaledHeight,'simple')
    if math.random() < flipProbability then
        input = image.hflip(input)
        label = image.hflip(label)
    end
    return input, label
end


function loadMiniBatch(batchSize, mode)
    if mode == 'train' then
        folderPath = dataPath .. 'train' .. '/'
    elseif mode == 'val' then
        folderPath = dataPath .. 'val' .. '/'
    elseif mode == 'test' then
        folderPath = dataPath .. 'test' .. '/'
    else
        print('Invalid Mode') 
    end
    local inputs = torch.Tensor(batchSize, 3, reScaledHeight, reScaledWidth)
    local labels = torch.Tensor(batchSize, 1, reScaledHeight, reScaledWidth)
    for i = 1, batchSize do
        input, label = loadInputAndLabel(folderPath)
        inputs[{{i},{},{},{}}] = input
        labels[{{i},{},{},{}}] = label
    end

    inputs = inputs[{{},{},{1,reScaledHeight},{1,reScaledWidth}}]
    labels = labels[{{},{},{1,reScaledHeight},{1,reScaledWidth}}]
    return inputs, labels
end



