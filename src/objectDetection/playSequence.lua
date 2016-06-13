require 'paths'
require 'image'
require 'qtwidget'

local path = '/mnt/data/resultVideos/sequence_04/images4/'

function playSequence()
    local input = image.load(path .. string.format('%06d.png',1))
    local rows = input:size(2)
    local cols = input:size(3)
    local win = qtwidget.newwindow(input:size(3), input:size(2), 'BB plotting')
    for i = 1,#paths.dir(path)-2 do
        sys.sleep(0.05)
        win:gbegin()
        local img = image.load(path .. string.format('%06d.png',i))
        image.display{image = img, win=win}
        win:gend()
    end
end


