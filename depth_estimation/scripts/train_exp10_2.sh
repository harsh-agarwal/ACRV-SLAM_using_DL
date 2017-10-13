#!/usr/bin/env sh

TOOLS=../../caffe/build/tools

$TOOLS/caffe train --solver solver_exp10_2.prototxt --snapshot /media/harsh/snapshots/exp10_2_iter_20000.solverstate  --gpu 0 --log_dir logs
  
  # run it from caffe/matlab/demo folder
