dir = require 'pl.dir';
require 'image';
require 'torch';
math.randomseed(os.time())

local dataPath = '/mnt/data/cityscapes/gtFine/train/'
local inputsPath = '/mnt/data/cityscapes/leftImg8bit/train/'

function mysplit(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; i = 1
    for str in string.gmatch(inputstr, "([^" .. sep .. "]+)") do
        t[i] = str
        i  = i + 1 
    end
    return t
end

function file_exists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end


function lines_from(file)
    if not file_exists(file) then 
        return {} 
    end
        local lines = {}
        for line in io.lines(file) do
            lines[#lines + 1] = line
        end
    return lines
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


function loadTestImage()
    local qtwidget  = require ('qtwidget')
    local directories = dir.getdirectories(dataPath)
    local indexFolder = math.random(1, #directories)
    local images, labels
    images = dir.getallfiles(dir.getdirectories(inputsPath)[indexFolder],'leftImg8bit.png')
    labels = dir.getallfiles(directories[indexFolder],'.txt')
    local imageIndex = math.random(1,#images)
    local input = image.load(images[imageIndex])
    local str = mysplit(images[imageIndex],'_')[2]
    input = image.scale(input,input:size(3)/2,input:size(2)/2, 'simple')
    local objects
    for i = 1, #labels do
        if mysplit(labels[i],'_')[2] == str then
            objects = lines_from(labels[i])
        end
    end
    local list = torch.zeros(#objects,4)
    for i = 1, #objects do
        list[i][1] = mysplit(objects[i],',')[2]/2
        list[i][2] = mysplit(objects[i],',')[4]/2
        list[i][3] = mysplit(objects[i],',')[3]/2
        list[i][4] = mysplit(objects[i],',')[5]/2
    end
    local win = qtwidget.newwindow(input:size(3), input:size(2), 'BB plotting')
    win:gbegin()
    win:showpage()
    image.display{image=input, win=win}
    draw_bb(list, 1, win, 'red')
end

    
