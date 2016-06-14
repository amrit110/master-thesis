dir = require 'pl.dir';
require 'image';
require 'torch';
math.randomseed(os.time())

local dataPath = '/mnt/data/KITTI_Object_Detection/'

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

local annotationsPath = dataPath .. 'training/annotations/'
local imagesPath = dataPath .. 'training/images/'
local annotations = dir.getallfiles(annotationsPath)
local images = dir.getallfiles(imagesPath)

local objectsList = {}
objectsList.cars = {}
objectsList.cars.details = {}
objectsList.cars.image = {}
objectsList.cars.annotations = {}
objectsList.pedestrians = {}
objectsList.pedestrians.details = {}
objectsList.pedestrians.image = {}
objectsList.pedestrians.annotations = {}
objectsList.cyclists = {}
objectsList.cyclists.details = {}
objectsList.cyclists.image = {}
objectsList.cyclists.annotations = {}


function makeObjectsList()
    for i = 1,#annotations do
        local objects = lines_from(annotations[i])
        local words = {}
        for j = 1,#objects do
            for word in objects[j]:gmatch("%S+") do 
                table.insert(words,word) 
            end
            if words[1] == 'Car' then
                table.insert(objectsList.cars.details,objects[j])
                table.insert(objectsList.cars.image,images[i])
                table.insert(objectsList.cars.annotations,annotations[i])
            elseif words[1] == 'Pedestrian' then
                table.insert(objectsList.pedestrians.details,objects[j])
                table.insert(objectsList.pedestrians.image,images[i])
                table.insert(objectsList.pedestrians.annotations,annotations[i])
            elseif words[1] == 'Cyclist' then
                table.insert(objectsList.cyclists.details,objects[j])
                table.insert(objectsList.cyclists.image,images[i])
                table.insert(objectsList.cyclists.annotations,annotations[i])
           end
        end
        xlua.progress(i,#annotations)
    end
    return objectsList, images
end









            
            

 
