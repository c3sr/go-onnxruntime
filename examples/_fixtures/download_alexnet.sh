#!/bin/sh

MODEL=torchvision_alexnet

mkdir $MODEL

wget http://s3.amazonaws.com/store.carml.org/models/onnxruntime/$MODEL.onnx -O $MODEL/$MODEL.onnx
wget http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt -O $MODEL/synset.txt