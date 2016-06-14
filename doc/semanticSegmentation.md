<a name="thesis.semanticSegmentation"></a>
## Semantic Segmentation ##
These  sections include a breif description of the usage of the code for the
semantic segmnentation implementation.

The implementation uses a pre-trained version of the VGG19 network, available
from the [loadcaffe](https://github.com/szagoruyko/loadcaffe) package. The path
to the pre-trained network is specified inside `model.lua`.

### Data set ###
The network is trained using the [Cityscapes data
set](https://www.cityscapes-dataset.com/). All images with fine annotations were renamed to the format
`00XXXX.png`, and corresponding lable files were created using [these](https://github.com/amrit110/cityscapesScripts)
convenient scripts. Be sure to set the correct path
variable inside `data.lua`:
```lua
dataPath = '/......../cityscapesProcessed/'
```


### Training ###
The script `main.lua` contains useful parameters for the network training,
stored in table `opt`. It
also declares a high-level training function `train()`.
To speciy if gradients should propagate through the entire network or only the
deconvolutional part of the network, use the function `setParametersNet()` with arguments
`'model'` or `'heads'`. 


### Testing ###
