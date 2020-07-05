#!/bin/sh

MODEL=torchvision_alexnet

mkdir $MODEL

wget http://s3.amazonaws.com/store.carml.org/models/onnxruntime/torchvision_alexnet.onnx -O $MODEL/torchvision_alexnet.onnx
wget http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt -O $MODEL/synset.txt