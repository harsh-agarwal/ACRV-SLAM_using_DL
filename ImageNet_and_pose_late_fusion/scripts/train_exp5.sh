#!/usr/bin/env sh

TOOLS=../../caffe/build/tools

$TOOLS/caffe train --solver solver_exp5.prototxt --weights ../bvlc_alexnet.caffemodel  --gpu 0 --log_dir logs
  
  # run it from caffe/matlab/demo folder
