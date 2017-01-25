# keras_nmt
This projects provides an nmt implementation with keras. Data parallel training also supported.
To run NMT experiments: 

CUDA_VISIBLE_DEVICES=2,3 DEVICES=/cpu:0,/gpu:0,/gpu:1 python train.py --state config.py > tf_80_2_gpu.log 2>&1 &

