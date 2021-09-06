#!/bin/bash

device_id="/gpu:0"
train_data="./data/modelnet10_train.h5"
test_data="./data/modelnet10_test.h5"
num_class=10
save_dir="out_modelnet10"
epoch_training=300
minibatch_size=16
num_embdim=256
num_point=1024
encoder_arch="PointNet"
# encoder_arch="DGCNN"
lmr_knn=20
lmr_lambda=1.0

command="python -u deepdiffusion_3dshape.py --train_data=$train_data --test_data=$test_data --num_class=$num_class --save_dir=$save_dir --device_id=$device_id --epoch_training=$epoch_training --minibatch_size=$minibatch_size --num_embdim=$num_embdim --num_point=$num_point --encoder_arch=$encoder_arch --lmr_knn=$lmr_knn --lmr_lambda=$lmr_lambda"
echo $command
$command # execute

exit
