local Tiling, parent = torch.class('nn.Tiling', 'nn.Module')

function Tiling:__init(...)
   parent.__init(self)
   local arg = {...}

   -- Takes only 1 argument: tile
   self.tile = arg[1]

   --[[
   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   if torch.type(arg[#arg]) == 'boolean' then
      self.batchMode = arg[#arg]
      table.remove(arg, #arg)
   end
   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n)
      for i=1,n do
         self.size[i] = arg[i]
      end
   end

   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
   end
   --]]
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


   --[[
   if (self.batchMode == false) or (
         (self.batchMode == nil) and
         (input:nElement() == self.nelement and input:size(1) ~= 1)
      ) then
      self.output:view(input, self.size)
   else
      self.batchsize[1] = input:size(1)
      self.output:view(input, self.batchsize)
   end
   --]]
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

   --[[
   self.gradInput:viewAs(gradOutput, input)
   --]]
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

