require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
collectgarbage()


local img = torch.rand(1,3,640,480):cuda()
local resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-34.t7')
-- Remove the fully connected layer
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
resnet:remove(#resnet.modules)
resnet:cuda()

local start = torch.tic()
out = resnet:forward(img)
local stop = torch.toc(start)
local resnet34Time = stop
collectgarbage()

local resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-34.t7')
-- Remove the fully connected layer
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
resnet:remove(#resnet.modules)
resnet:cuda()

local start = torch.tic()
out = resnet:forward(img)
local stop = torch.toc(start)
local resnet34Time = stop
collectgarbage()



local resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-18.t7')
-- Remove the fully connected layer
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
resnet:remove(#resnet.modules)
resnet:cuda()

local start = torch.tic()
out = resnet:forward(img)
local stop = torch.toc(start)
local resnet18Time = stop
collectgarbage()




local resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-101.t7')
-- Remove the fully connected layer
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.Linear')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'nn.View')
resnet:remove(#resnet.modules)
assert(torch.type(resnet:get(#resnet.modules)) == 'cudnn.SpatialAveragePooling')
resnet:remove(#resnet.modules)
resnet:cuda()

local start = torch.tic()
out = resnet:forward(img)
local stop = torch.toc(start)
local resnet101Time = stop
collectgarbage()


local VGG = torch.load('/mnt/data/pretrainedModels/VGG/VGG.t7'):cuda()

local start = torch.tic()
out = VGG:forward(img)
local stop = torch.toc(start)
local VGGTime = stop
collectgarbage()


local inception = torch.load('/mnt/data/pretrainedModels/inceptionv3/inceptionv3.t7'):cuda()

local start = torch.tic()
out = inception:forward(img)
local stop = torch.toc(start)
local inceptionTime = stop
collectgarbage()



print('ResNet-18 time - ' ..  string.format('%.3f',resnet18Time))
print('ResNet-34 time - ' ..  string.format('%.3f',resnet34Time))
print('ResNet-101 time - ' ..  string.format('%.3f',resnet101Time))
print('VGG time - ' ..  string.format('%.3f',VGGTime))
print('Inception time - ' ..  string.format('%.3f',inceptionTime))







