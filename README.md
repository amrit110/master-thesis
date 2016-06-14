# Master Thesis

This repository contains a the code used during the master thesis project
"Vehicle detection and Road Scene Segmentation using Deep Learning" (Link to
report). All approaches are based on Convolutional Neural Networks (CNNs) and are
implemented in LuaJIT using [Torch7](http://torch.ch/).  More specifically, the repository includes

* An [MNIST-based small scale detection](doc/mnist_detection.md) implementation. A CNN is 
trained to classify regions of an image with spread out [MNIST](http://yann.lecun.com/exdb/mnist/) digits.
Bounding boxes for digits are achieved through regression. The task was
implemented as a proof if concept for the "detection through classification"
approach used for vehicle detection.

* A [vehicle detection](doc/objectDetection.md) implementation trained on the 
[KITTI object detection data set](http://www.cvlibs.net/datasets/kitti/eval_object.php).
This was the main focus for the thesis.

* An implementation of a network trained for [semantic
  segmentation](doc/semanticSegmentation.md) based on a [deconvolutional
approach](http://arxiv.org/abs/1505.04366) by Noh et al.

