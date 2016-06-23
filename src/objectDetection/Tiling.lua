local Tiling, parent = torch.class('nn.Tiling', 'nn.Module')

function Tiling:__init(...)
   parent.__init(self)
   local arg = {...}

   -- Takes only 1 argument: tile
   self.tile = arg[1]
end

function Tiling:updateOutput(input)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end
   -- ONLY WORKING IN BATCH MODE:
   local batchSize = input:size(1)
   local channelsIn = input:size(2)
   assert(channelsIn % (self.tile*self.tile) == 0,'Number of input channels: ' .. channelsIn .. ' not divisable by square of tile: ' .. self.tile*self.tile)
   local channelsOut = channelsIn/(self.tile*self.tile)
   local width = input:size(3)
   local height = input:size(4)

   -- Resize output to correct size
   self.output:resize(batchSize, channelsOut, width*self.tile, height*self.tile)

   -- Loop over every spatial loocation of the input
   for i = 1, width do
       for j = 1, height do
            local block = input[{{},{},{i},{j}}]:clone():view(batchSize, channelsOut, self.tile, self.tile)
            self.output[{{},{},{(i-1)*self.tile+1, i*self.tile},{(j-1)*self.tile+1, j*self.tile}}] = block
       end
   end

   return self.output
end

function Tiling:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   -- ONLY WORKING IN BATCH MODE:
   local batchSize = input:size(1)
   local channelsIn = input:size(2)
   assert(channelsIn % (self.tile*self.tile) == 0,'Number of input channels: ' .. channelsIn .. ' not divisable by square of tile: ' .. self.tile*self.tile)
   local channelsOut = channelsIn/(self.tile*self.tile)
   local width = input:size(3)
   local height = input:size(4)

   -- Resize output to correct size
   self.gradInput:resize(input:size())

   -- Loop over every spatial loocation of the input
   for i = 1, width do
       for j = 1, height do
            local block = gradOutput[{{},{},{(i-1)*self.tile+1, i*self.tile},{(j-1)*self.tile+1, j*self.tile}}]:clone():view(batchSize, channelsIn, 1, 1)
            self.gradInput[{{},{},{i},{j}}] = block
       end
   end

   return self.gradInput
end


function Tiling:__tostring__()
  return torch.type(self) .. '(' ..
      table.concat({self.tile}, 'x') .. ')'
end

function Tiling:clearState()
   nn.utils.clear(self, '_input', '_gradOutput')
   return parent.clearState(self)
end

