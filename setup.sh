#!/bin/bash

data_dir="./data"
training_dir="./data/training"
test_dir="./data/test"

mkdir -p $training_dir $test_dir
cd $data_dir
wget -i ../MNIST_urls.txt
mv -t ./training train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
mv -t ./test t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
