require 'image'
require 'nn'
require 'torch'
require 'qt'
require 'qtwidget'

-- Plot classifier output when feeding a 72x72 sample
function plot_classifier(indx)
    local matio = require 'matio'
    local indx = indx or 3
    local testset = torch.load(path .. 'data/mnist_72_test.t7')
    local img = testset.data[indx]
    local label = testset.label[indx]
    local label_old = testset.label_old[indx]

    local zpdNet = nn.Sequential():add(nn.SpatialZeroPadding(11,11,11,11))
    local imgZpd = zpdNet:forward(img):cuda()
    --imgZpd = nn.Unsqueeze(1):forward(imgZpd):cuda()

    -- Classifier output:
    local out = model:forward(imgZpd)[1]:float():squeeze()
    local softNet = nn.Sequential():add(nn.SpatialSoftMax())
    local pred = softNet:forward(out)
    image.display{image=pred, zoom=4, padding=1}
    image.display{image=img, zoom=4}
    if false then
        matio.save('heatmaps.mat',pred)
        matio.save('scatter.mat',img)
    end
    if false then
        upsampNet = nn.SpatialUpSamplingNearest(10)
        pred = upsampNet:forward(pred)
        for i=1,11 do
            image.save('images/' .. tostring(i) .. '.png',pred[i])
        end
        img = img:add(-img:min())
        img = img:div(img:max())
        img = upsampNet:forward(img:float())
        image.save('images/input.png', img)
        --label_old = label_old:add(-label_old:min())
        label_old[label_old:eq(11)] = 5
        label_old = label_old:div(label_old:max())
        label_old = upsampNet:forward(nn.Unsqueeze(1):forward(label_old:float()))
        image.save('images/label_old.png', label_old)
    end
end

-- Plot bounding boxes for a set of 26x26 samples
function plot_bb(indx,net_class_bb)
    local testset = torch.load(path .. 'data/mnist_test.t7')
    testset.data = testset.data:float()
    testset.label = testset.label:float()
    local n_img = 30
    local indx = indx or 1
    local img = testset.data[{{indx,indx + n_img - 1}}]
    local lab = testset.label[{{indx,indx + n_img - 1}}]
    lab = lab[{{},{2,5}}]
    local out_bb
    if net_class_bb then
        out_bb = torch.Tensor(30,4)
        for i=1,n_img do
            out_bb[i] = model:forward(img[i])[2]:squeeze()
        end
    end

    -- PLOTTING
    local win_zoom = 6
    local win = qtwidget.newwindow(6*26*win_zoom, 3*26*win_zoom, 'MNIST detection')
    win:gbegin()
    win:showpage()
    image.display{image=img, win=win, zoom=win_zoom}

    local function offset_6x5(list)
        for i=1,n_img do
            local offset = torch.Tensor{26*((i-1)%6), 26*math.floor((i-1)/6),
                                        26*((i-1)%6), 26*math.floor((i-1)/6)}
            list[i]:add(offset)
        end
        return list
    end

    draw_bb(offset_6x5(lab), win_zoom, win, 'blue')
    if net_class_bb then
        draw_bb(offset_6x5(out_bb:squeeze()), win_zoom, win, 'green')
    end
end


-- Plot threasholded bounding boxes given a 72x72 sample
function plot_inference(class, indx, threshold)
    -- LOADING
    local testset = torch.load(path .. 'data/mnist_72_test.t7')
    local indx = indx or 7
    local threshold = threshold or 0.95
    local img = testset.data[indx]
    local lab = testset.label[indx]

    -- RUN NETWORK
    local zpdNet = nn.Sequential():add(nn.SpatialZeroPadding(11,11,11,11))
    local imgZpd = zpdNet:forward(img):cuda()

    local out = model:forward(imgZpd)
    local out_c = nn.SpatialSoftMax():forward(out[1]:float())
    local out_bb = out[2]:float()
    local nRows = out_c:size(2)
    local nCols = out_c:size(3)

    -- COORDINATE TRANSLATION
    local r = torch.range(0,17)*4 -- x4 due to output resolution is 1/4
    out_bb[{{1},{},{}}]:add(r:view(1,1,18):expandAs(out_bb[{{1},{},{}}]))
    out_bb[{{2},{},{}}]:add(r:view(1,18,1):expandAs(out_bb[{{1},{},{}}]))
    out_bb[{{3},{},{}}]:add(r:view(1,1,18):expandAs(out_bb[{{1},{},{}}]))
    out_bb[{{4},{},{}}]:add(r:view(1,18,1):expandAs(out_bb[{{1},{},{}}]))
    out_bb:add(-11) -- context view is 26x26, activation area is 4x4 -> zpd = 11
    -- restrict BB within image:
    out_bb = nn.ReLU():forward(out_bb)
    out_bb = nn.ReLU():forward(-out_bb + 72)
    out_bb = -out_bb + 72
    
    -- PLOTTING
    local win_zoom = 6
    local win = qtwidget.newwindow(img:size(3)*win_zoom, img:size(2)*win_zoom, 'MNIST detection')
    win:gbegin()
    win:showpage()
    image.display{image=img, win=win, zoom=win_zoom}
    
    -- Loop over classes
    classes = {5,6,8,10}
    colors = {{1,0,0},{0,1,0},{0,0,1},{1,1,0}}
    for i, class in ipairs(classes) do
        for row=1,nRows do
            for col = 1,nCols do
                if out_c[{{class},{row},{col}}]:squeeze() > threshold then
                    local coord = out_bb[{{},{row},{col}}]:clone()
                    draw_bb(coord:view(1,4), win_zoom, win, colors[i])
                end
            end
        end
    end
    ----
    return win
end


-- Help function to draw bounding boxes
function draw_bb(list, win_zoom, win, color)
    local n_img = list:size(1)
    for i = 1,n_img do
        local coords = list[i]:squeeze():totable()
        local w = (coords[3]-coords[1])*win_zoom
        local h = (coords[4]-coords[2])*win_zoom
        local x = (coords[1])*win_zoom
        local y = (coords[2])*win_zoom

        win:setcolor(color[1],color[2],color[3],0.2)
        win:setlinewidth(2)
        win:rectangle(x,y,w,h)
        win:stroke()
        win:gend()
    end
end
