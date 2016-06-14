<a name="thesis.mnist"></a>
## MNIST detection ##

These sections includes a breif description of the basic usage for the MNIST detection
implementation.

### Data sets ###
The implementation uses a synthetic dataset created from the original MNIST
images. It is created by running:
```bash
$ th create_datasets_72.lua
$ th extract crops.lua
```
This will download the original MNIST dataset, create files `mnist_72.t7` and
`mnist_72_test.t7`, extract crops from
those and store them as `mnist_train.t7` and `mnist_test.t7`.

### Training ###
The script `main_cuda.lua` contains all parameters  used to declare the training
and a high-level training function `train()`. Make sure to set up the absolute path to the repository:
```lua
path = '/......./masterThesis/src/mnist_detection/'
```
### Testing ###
The network accuracy is tested with the `evaluateError()` function, taking
arguments `'validation'` or `'test'`. Training cost can be plotted with function
`plotCost()`.

The file `plotfunc_cuda.lua` contains various functions used to test the
network performance qualitatively.


