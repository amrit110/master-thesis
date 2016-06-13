cv = require 'cv'
require 'cv.objdetect'

function groupBoxes(boxes, groupThreshold, eps)
    local boxCopy = boxes:clone()
    boxCopy[{{},{1}}]:add(boxes[{{},{3}}]):div(2)
    boxCopy[{{},{2}}]:add(boxes[{{},{4}}]):div(2)
    boxCopy[{{},{3}}]:csub(boxes[{{},{1}}])
    boxCopy[{{},{4}}]:csub(boxes[{{},{2}}])
    local rectTable = {}
    for i=1,boxCopy:size(1) do
        local r = cv.Rect(boxCopy[{i,1}],boxCopy[{i,2}],boxCopy[{i,3}],boxCopy[{i,4}])
        table.insert(rectTable,r)
    end
    local rectArray = cv.newArray('cv.Rect', rectTable)
    local boxGrouped, t = cv.groupRectangles{rectList = rectArray, groupThreshold = groupThreshold, eps = eps}
    local newBoxes = boxes.new():resize(boxGrouped.size,4)
    for i=1,boxGrouped.size do
        local w = boxGrouped.data[i].width
        local h = boxGrouped.data[i].height
        newBoxes[{i,3}] = boxGrouped.data[i].x + w/2
        newBoxes[{i,4}] = boxGrouped.data[i].y + h/2
        newBoxes[{i,1}] = boxGrouped.data[i].x - w/2
        newBoxes[{i,2}] = boxGrouped.data[i].y - h/2
    end
    return newBoxes
end
