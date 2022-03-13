#!/bin/bash
#This is a multinode runscript to run on 2 nodes with 4 GPUs each

export PYTHON=~/anaconda3/bin/python
export PROGRAM=~/submission_code/main.py #Assuming that the program is unzipped in the /home/$USER folder
export DATADIR=~/data
export TOTALGPUS=2
export GPUSPERNODE=1
export MPIRUN=mpirun
export LRMBSGDB=0.8
export LRMBSGDBF=3.2
export LRHWB=0.4
export LRPHWB=0.5
export HOST=10.0.1.41 #10.36.192.244
export SECONDHOST=10.0.1.42 #10.36.192.245
# Add the IP addresses of other hosts if needed
# Accordingly Change the total number of GPUs and GPUs per node variables TOTALGPUS and GPUSPERNODE respectively


mkdir -p wnet168cifar10_multi_node
cd wnet168cifar10_multi_node

######################LAPSGD##################################

$MPIRUN -n $TOTALGPUS -H $HOST:$GPUSPERNODE,$SECONDHOST:$GPUSPERNODE -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model wnet168 \
--cuda --train_processing_bs 64 --test_processing_bs 64 --lr $LRHWB \
--baseline_lr 0.05 --weight-decay 0.0001 --seed 6 --pm  \
--nesterov --workers 0 --num-threads 4 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 100 --scheduler-type cosine \
--training-type LAPSGD \
--bs_multiple 1 --test_bs_multiple 1 --epochs 200 --averaging_freq 16 \
--warm_up_epochs 5 --dist-url tcp://$HOST:23456  --storeresults

##################LPPSGD#######################

$MPIRUN -n $TOTALGPUS -H $HOST:$GPUSPERNODE,$SECONDHOST:$GPUSPERNODE -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model wnet168 \
--cuda --train_processing_bs 64 --test_processing_bs 64 --lr $LRPHWB \
--baseline_lr 0.05 --weight-decay 0.0001 --seed 6 --pm  \
--nesterov --workers 0 --num-threads 4 --test-freq 1 --partition \
--num-processes 3 --pre_post_epochs 100 --scheduler-type cosine \
--training-type LPPSGD  --prepassmepochs 20 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 200 --averaging_freq 16 \
--warm_up_epochs 5 --dist-url tcp://$HOST:23456  --storeresults

######################PLSGD##################################
$MPIRUN -n $TOTALGPUS -H $HOST:$GPUSPERNODE,$SECONDHOST:$GPUSPERNODE -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model wnet168 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr $LRMBSGDB \
--baseline_lr 0.1 --weight-decay 0.0001 --seed 6 --pm  \
--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 100 --scheduler-type mstep \
--training-type PLSGD --lrmilestone 60 120 160 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 200 --averaging_freq 16 \
--warm_up_epochs 5 --dist-url tcp://$HOST:23456  --storeresults

##################MBSGD##############################

$MPIRUN -n $TOTALGPUS -H $HOST:$GPUSPERNODE,$SECONDHOST:$GPUSPERNODE -bind-to none -map-by slot \
$PYTHON $PROGRAM --data-dir $DATADIR  \
--dataset cifar10 --num-classes 10 --momentum 0.9 --model wnet168 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr $LRMBSGDB \
--baseline_lr 0.1 --weight-decay 0.0001 --seed 6 --pm  \
--nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
--pre_post_epochs 100 --scheduler-type mstep --gamma 0.2 \
--training-type MBSGD --lrmilestone 60 120 160 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 200 \
--warm_up_epochs 5 --dist-url tcp://$HOST:23456  --storeresults

cd ..
