#!/bin/bash
neural_ode_norm_dir="/workspace/home/jgusak/neural-ode-norm"

data_root="$neural_ode_norm_dir/data/"
save_dir="$neural_ode_norm_dir/save/"
train_file="$neural_ode_norm_dir/train.py"

config="$neural_ode_norm_dir/configs/odenet4_bn-ln_euler-32.cfg"
source $config

bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python3 $train_file \
                 --data_root $data_root \
                 --network $network \
                 --batch_size $batch_size \
                 --lr $lr\
                 --save $save_dir\
                 --method $solver \
                 --n_nodes $n_nodes \
                 --inplanes $inplanes \
                 --normalization_resblock $normalization_resblock\
                 --normalization_odeblock $normalization_odeblock\
                 --normalization_bn1 $normalization_bn1\
                 --param_normalization_resblock $param_normalization_resblock\
                 --param_normalization_odeblock $param_normalization_odeblock\
                 --param_normalization_bn1 $param_normalization_bn1\
                 --activation_resblock $activation_resblock\
                 --activation_odeblock $activation_odeblock\
                 --activation_bn1 $activation_bn1\
                 --num_epochs $num_epochs\
                 --save_every $save_every\
                 --torch_seed $torch_seed"