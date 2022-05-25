#!/bin/bash
#This is a singlenode runscript to run on 2 GPUs each

ml fosscuda/2019a 
# export PYTHON=~/miniconda3/bin/python
export PYTHON=~/anaconda3/bin/python
export PROGRAM=~/workspace/LPP-SGD/main.py #Assuming that the program is unzipped in the /home/$USER folder
export TOTALGPUS=2 # Change the total number of GPUs if needed
# export MPIRUN=/usr/bin/mpirun
export DATADIR=~/workspace/data
export MPIRUN=mpirun #/mnt/appl/software/OpenMPI/4.0.3-gcccuda-2020a/bin/mpirun
mkdir -p rn20cifar10_single_node
cd rn20cifar10_single_node

###########################LAPSGD#####################################

$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.1 \
--baseline_lr 0.1 --weight-decay 0.0005 --seed 6 --pm  \
--nesterov --workers 0 --num-threads 2 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 150 --scheduler-type cosine \
--training-type LAPSGD \
--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --averaging_freq 16 \
--warm_up_epochs 5 --dist-url tcp://localhost:23456  --storeresults

############################LPPSGD#####################################
#
$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.125 \
--baseline_lr 0.1 --weight-decay 0.0005 --seed 6 --pm  \
--nesterov --workers 0 --num-threads 2 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 150 --scheduler-type cosine \
--training-type LPPSGD  --prepassmepochs 30 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --averaging_freq 16 \
--warm_up_epochs 5 --dist-url tcp://localhost:23456  --storeresults
#
############################PLSGD#####################################
#
$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
--cuda --train_processing_bs 1024 --test_processing_bs 1024 --lr 0.8 \
--baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
--scheduler-type mstep --training-type PLSGD  --pre_post_epochs 150 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --warm_up_epochs 5 \
--dist-url tcp://localhost:23456 --lrmilestone 150 225  --storeresults --lars


$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.1 \
--baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
--scheduler-type mstep --training-type PLSGD  --pre_post_epochs 150 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --warm_up_epochs 5 \
--dist-url tcp://localhost:23456 --lrmilestone 150 225  --storeresults --lars
#
############################MBSGD#####################################
#
# $MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
# $PYTHON $PROGRAM --data-dir $DATADIR  \
# --dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
# --cuda --train_processing_bs 1024 --test_processing_bs 1024 --lr 0.8 \
# --baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --scheduler-type mstep --training-type MBSGD \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 300 --warm_up_epochs 5 \
# --dist-url tcp://localhost:23456 --lrmilestone 150 225  --storeresults --lars
#
############################PLSGD+LARS#####################################
#
#$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
#$PYTHON $PROGRAM --data-dir $DATADIR  \
#--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
#--cuda --train_processing_bs 1024 --test_processing_bs 1024 --lr 0.8 \
#--baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
#--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
#--scheduler-type mstep --training-type PLSGD  --pre_post_epochs 150 \
#--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --warm_up_epochs 5 \
#--dist-url tcp://localhost:23456 --lrmilestone 150 225  --storeresults --lars
#
############################MBSGD+LARS#####################################
#
#$MPIRUN -n $TOTALGPUS -H localhost:$TOTALGPUS -bind-to none -map-by slot \
#$PYTHON $PROGRAM --data-dir $DATADIR  \
#--dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
#--cuda --train_processing_bs 1024 --test_processing_bs 1024 --lr 0.8 \
#--baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
#--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
#--scheduler-type mstep --training-type MBSGD \
#--bs_multiple 1 --test_bs_multiple 1 --epochs 300 --warm_up_epochs 5 \
#--dist-url tcp://localhost:23456 --lrmilestone 150 225  --storeresults --lars
#
cd ..
