#!/bin/bash
#This is a slurm managed runscript to run on 8 GPUs, one on each node

#!/bin/bash
#SBATCH --nodes=8               # number of nodes
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=24       # number of CPU cores per process
#SBATCH --gres=gpu:1            # GPUs per node
#SBATCH --hint=compute_bound
#SBATCH --hint=multithread
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=my_test        # job name (default is the name of this file)

ml fosscuda/2019a #Load the modules for mpirun, etc. 

PYTHON=~/anaconda3/bin/python
PROGRAM=~/submission_code/main.py  #Assuming that the program is unzipped in the /home/$USER folder
DATADIR=~/data
IMAGENETDIR=~/ILSVRC
MASTER=`/bin/hostname -s`

mkdir -p rnet50Imagenet_Slurm
cd rnet50Imagenet_Slurm

########################LPPSGD#####################################

srun \
$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
--cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.125 \
--baseline_lr 0.0125 --weight-decay 0.00005 --seed 42 --pm \
--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 45 \
--scheduler-type cosine \
--training-type LPPSGD  --prepassmepochs 6 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 8 \
--warm_up_epochs 5 --dist-url tcp://$MASTER:23456  \
--numnodes $SLURM_JOB_NUM_NODES --storeresults

########################LAPSGD#####################################

srun \
$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
--cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
--baseline_lr 0.0125 --weight-decay 0.00005 --seed 42 --pm \
--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 45 \
--scheduler-type cosine \
--training-type LAPSGD \
--bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 8 \
--warm_up_epochs 5 --dist-url tcp://$MASTER:23456  \
--numnodes $SLURM_JOB_NUM_NODES --storeresults


########################MBSGD#####################################

srun \
$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
--baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
--scheduler-type mstep --lrmilestone 30 60 80 --pre_post_epochs 45 \
--training-type MBSGD --numnodes $SLURM_JOB_NUM_NODES \
--bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
--warm_up_epochs 5 --dist-url tcp://$HOSTNAME:23456  --storeresults

########################PLSGD#####################################

srun \
$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
--baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
--scheduler-type mstep --lrmilestone 30 60 80 --pre_post_epochs 45 \
--averaging_freq 8 --numnodes $SLURM_JOB_NUM_NODES \
--training-type PLSGD --pre_post_epochs 45 \
--bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
--warm_up_epochs 5 --dist-url tcp://$MASTER:21456  --storeresults



cd ..