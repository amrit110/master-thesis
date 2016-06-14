require 'torch'
require 'image'
--[[
This script loads the datasets 'mnist_72.t7' and 'mnist_72_test.t7'.
For each digit in each dataset a crop of 26x26 is generated.
The crop is offcenterd by some maximum value.
Then, 0-3 crops are extracted from the background.
The labels are stored as
[class, x1, y1, x2, y2],
where the coordinates are given relative to the upper left corner
of the 26x26 images.
The datasets are stored in 'mnist_train.t7' and 'mnist_test.t7'
--]]

-- Parameters
img_size = 72
crop_size = 26
max_offset = 5
overlap_threshold = 0.8
label_old_size = 72
background_class = 11

function create_26(dset)
    local n_72 = #dset.label
    -- 5 crops from each 72x72 image
    local n_26 = n_72*5
    local dset_new = {}
    dset_new.data = torch.Tensor(n_26,1,crop_size,crop_size)
    -- Label stored as (class, x1, y1, x2, y2)
    dset_new.label = torch.Tensor(n_26, 5)

    function getObjectCrop(img,coordinates)
        local cx = (coordinates[1]+coordinates[3])/2
        local cy = (coordinates[2]+coordinates[4])/2
        cx = cx + math.random(-max_offset,max_offset)
        cy = cy + math.random(-max_offset,max_offset)
        cx = math.max(crop_size/2+1, math.min(img_size-crop_size/2+1,cx))
        cy = math.max(crop_size/2+1, math.min(img_size-crop_size/2+1,cy))
        local x1 = math.floor(cx - crop_size/2)
        local x2 = math.floor(cx + crop_size/2-1)
        local y1 = math.floor(cy - crop_size/2)
        local y2 = math.floor(cy + crop_size/2-1)
        local crop = img[{{},{y1,y2},{x1,x2}}]
        box = torch.Tensor(4)
        box[1] = coordinates[1] - x1
        box[2] = coordinates[2] - y1
        box[3] = coordinates[3] - x1
        box[4] = coordinates[4] - y1
        return crop, box
    end

    function getBackgroundCrop(img,labels_old)
        local valid_background = 0
        local crop
        repeat
            local cx = math.random(1,img_size)
            local cy = math.random(1,img_size)
            cx = math.max(crop_size/2+1, math.min(img_size-crop_size/2+1,cx))
            cy = math.max(crop_size/2+1, math.min(img_size-crop_size/2+1,cy))
            local x1 = math.floor(cx - crop_size/2)
            local x2 = math.floor(cx + crop_size/2-1)
            local y1 = math.floor(cy - crop_size/2)
            local y2 = math.floor(cy + crop_size/2-1)
            crop = img[{{},{y1,y2},{x1,x2}}]
            label_crop = labels_old[{{y1,y2},{x1,x2}}]
            threshold = torch.sum(label_crop:eq(background_class))/(crop_size*crop_size) 
            if threshold > overlap_threshold then
                valid_background = 1
            end
        until valid_background == 1
        return crop
    end

    -- Main loop:
    for i=1,n_72 do
        local img = dset.data[i]
        local labels = dset.label[i]
        local labels_old = dset.label_old[i]
        local n_digits = labels:size(1)
        for j=1,5 do
            if j <= n_digits then
                -- Draw object sample
                local lab = labels[j]
                local class = lab[1]
                local coordinates = lab[{{2,5}}]
                local crop, box = getObjectCrop(img,coordinates)
                dset_new.data[(i-1)*5+j] = crop
                dset_new.label[{{(i-1)*5+j},{1}}] = class
                dset_new.label[{{(i-1)*5+j},{2,5}}] = box
            else
                -- Draw backgrownd sample
                local crop = getBackgroundCrop(img,labels_old)
                local box = torch.Tensor{-1,-1,-1,-1}
                dset_new.data[(i-1)*5+j] = crop
                dset_new.label[{{(i-1)*5+j},{1}}] = background_class
                dset_new.label[{{(i-1)*5+j},{2,5}}] = box
            end
        end
    end


    local new_set = {}
    new_set.data = dset_new.data
    new_set.label = dset_new.label

    -- Give data set some properties, and normalize it
    setmetatable(new_set,
        {__index = function(t,i)
            return {t.data[i], t.label[i]}
        end}
    );
    new_set.data = new_set.data:double()
    new_set.label = new_set.label:double()
    function new_set:size()
        return self.data:size(1)
    end
    local mean = new_set.data:mean()
    new_set.data:add(-mean)
    local stdv = new_set.data:std()
    new_set.data:div(stdv)
    return new_set
end

training_72 = torch.load('mnist_72.t7')
test_72 = torch.load('mnist_72_test.t7')
training_26 = create_26(training_72)
test_26 = create_26(test_72)
torch.save('mnist_train.t7',training_26)
torch.save('mnist_test.t7',test_26)
