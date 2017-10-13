#!/usr/bin/env sh

TOOLS=../../caffe/build/tools

$TOOLS/caffe train --solver solver_exp14_3.prototxt  --weights /media/harsh/snapshots/exp14_3_iter_6000.caffemodel --gpu 1 --log_dir logs
  
  # run it from caffe/matlab/demo folder
