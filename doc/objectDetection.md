<a name="thesis.objectDetection"></a>
## Object Detection ##

These  sections include a breif description of the usage of the code for the
vehicle detection implementation. The CNN based implementation has three heads
used for classification, bounding boxes and range estimation.

The implementation uses a pre-trained version of the Resnet-34, available
[here](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained). Be sure to set the correct path to the
pre-trained model inside the function `initialiseModel()` in
`model_resnet.lua`.



### Data set ###
The network is trained usig the [KITTI object detection data
set](http://www.cvlibs.net/datasets/kitti/eval_object.php),
and uses multiple threads (donkeys) to load the images from disk with functions
declared in `donkeyCrops.lua`. Make sure to set the correct path to the
data set:
```lua
local dataPath = '/......./KITTI_Object_Detection/'
```
Same goes for the global `opt.path` variable, declared in `main.lua`.

### Training ###
The script `main.lua` contains useful parameters for the network training,
stored in table `opt`. It
also declares a high-level training function `train()`. The parameter
`opt.criterionWeights` is used to balance the cost functions used for the three
heads and can be set to 0 if any of them should be ignored during training. To
speciy if gradients should propagate through the entire network or only the
heads of the network, use the function `setParameterNetwork()` with arguments
`'model'` or `'heads'`. 


### Testing ###
