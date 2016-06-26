--script to load a pretrained resnet model and dilate the kernels to be used
--for semantic segmentation. supports only low resolution images so the data
--file to be modified accordingly

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'


function initialiseModel()
-- Limitations: will only dilate 3x3 kernels
    
    resnet = torch.load('/mnt/data/pretrainedModels/resnet/resnet-18.t7')
    function setDilation(index, zeroPadding, dilation)
        local subnet = resnet
        local lastIndex = index[#index]
        index[#index] = nil
        for i, v in pairs(index) do
            subnet = subnet.modules[v]
        end
        local inModule = subnet.modules[lastIndex]

        local nInputPlane = inModule.nInputPlane
        local nOutputPlane = inModule.nOutputPlane
        local weights = inModule.weight
        local bias = inModule.bias
        local outModule = nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane,3,3,1,1,zeroPadding,zeroPadding, dilation, dilation)
        outModule.weight = weights:clone()
        outModule.bias = bias:clone()

        subnet:remove(lastIndex)
        subnet:insert(outModule, lastIndex)
    end

    function removeStride(index,subnet)
        local subnet = resnet
        for i, v in pairs(index) do
            subnet = subnet.modules[v]
        end
        local inModule = subnet

        inModule.dW = 1
        inModule.dH = 1
    end

--removeStride({1})
--setDilation({5,1,1,1,1}, 4, 4)
    removeStride({6,1,1,1,1})
    setDilation({6,1,1,1,4}, 2, 2)
    removeStride({6,1,1,2})

    removeStride({7,1,1,1,1})
    setDilation({7,1,1,1,4}, 2, 2)
    removeStride({7,1,1,2})

    removeStride({8,1,1,1,1})
    setDilation({8,1,1,1,4}, 2, 2)
    removeStride({8,1,1,2})

    -- Remove 7x7 avg pooling, view, linear
    resnet:remove(11)
    resnet:remove(10)
    resnet:remove(9)
    -- Set max pooling layer kernel width to 2 = stride
    resnet.modules[4] = nn.SpatialMaxPooling(2,2,2,2)

    resnet:add(nn.SpatialMaxUnpooling(resnet.modules[4]))
    resnet:add(nn.SpatialFullConvolution(512,nClasses, 7,7,2,2,3,3,1,1))
    return resnet
end
