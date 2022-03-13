import argparse
import os
import glob
# import json
import shutil
import torch
import datetime
import numpy as np
# import torch.backends.cudnn as cudnn
import platform
import sys
sys.path.insert(0, "..")


def prepare_dir(args):
    x = datetime.datetime.now()
    datetimestring = str(x.hour) +\
        str(x.minute) + str(x.second) + "_" +\
        str(x.day) + str(x.month) + str(x.year)
    args.hpstring = args.training_type.title() + args.scheduler_type.title() +\
        args.model.title() + args.dataset.title() + \
        "Bs"+str(args.train_bs) + "ProcBs" + str(args.train_processing_bs) + \
        "Lr"+str(args.lr-np.fix(args.lr))[2:] + "AvFreq" + \
        str(args.averaging_freq) + "NumNodes" + str(args.numnodes)

    if 'HW' in args.training_type:
        args.hpstring = args.hpstring + 'Pr' + str(args.num_processes)
    if args.training_type in ['alPHW']:
        args.hpstring = args.hpstring + 'PsmSt' + str(args.assmswitchepochs[0])
    if args.cuda:
        args.hpstring = args.host + "GPU" + str(args.gpus[0]) + args.hpstring
    else:
        args.hpstring = args.host + args.hpstring
    curr_dir = os.getcwd()
    args.snap_dir = curr_dir + '/snapshots' + args.hpstring + "_" +\
        datetimestring
    args.results_dir = '/debug' if args.debug else '/results'
    args.results_dir = curr_dir + args.results_dir + args.hpstring +\
        "_" + datetimestring
    if args.clean_start:
        result_dirs = glob.glob('*' + args.hpstring + '*')
        for r in result_dirs:
            try:
                shutil.rmtree(r, ignore_errors=True)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)


def prepare_run_files(args):
    if not os.path.exists(args.results_dir + '/rundir'):
        os.makedirs(args.results_dir + '/rundir')
    runtbfile = open(args.results_dir + '/rundir/tb_start.sh', "w")
    runtbfile.write("kill $(lsof -t -i:6006)\n")
    runtbfile.write("tensorboard --bind_all --logdir=" + args.results_dir +
                    " --port=6006 &\n")
    runtbfile.close()
    runtbstopfile = open(args.results_dir + '/rundir/tb_stop.sh', "w")
    runtbstopfile.write("kill $(lsof -t -i:6006)\n")
    runtbstopfile.close()
    flaskfile = open(args.results_dir + '/rundir/app.py', "w")
    flaskfile.write("from flask import Flask\n")
    flaskfile.write("from flask import render_template\n")
    flaskfile.write("app = Flask(__name__)\n")
    flaskfile.write("@app.route(\"/\")\n")
    flaskfile.write("def hello():return render_template('index.html')\n")
    flaskfile.write("if __name__ == \"__main__\":app.run(debug=True)")
    flaskfile.close()
    flaskstartfile = open(args.results_dir + '/rundir/flask_start.sh', "w")
    flaskstartfile.write("kill $(lsof -t -i:5000)\n")
    flaskstartfile.write("mkdir " + args.results_dir + "/rundir/templates\n")
    flaskstartfile.write("cp " + args.results_dir + "/result.html " +
                         args.results_dir + "/rundir/templates/index.html\n")
    flaskstartfile.write("cd " + args.results_dir + "/rundir\n")
    flaskstartfile.write("python " + args.results_dir + "/rundir/app.py &\n")
    flaskstartfile.write("cd -\n")
    flaskstartfile.close()
    flaskstopfile = open(args.results_dir + '/rundir/flask_stop.sh', "w")
    flaskstopfile.write("kill $(lsof -t -i:5000)\n")
    flaskstopfile.write("rm -r " + args.results_dir + "/rundir/templates\n")
    flaskstopfile.close()


def filter_args(args):
    args.cuda = args.cuda and torch.cuda.is_available()
    if not args.cuda:
        args.pm = False
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.commrank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.commsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_PROCID' in os.environ:
        args.commrank = int(os.environ['SLURM_PROCID'])
        args.commsize = int(os.environ['SLURM_NTASKS'])

    if args.commrank != 0:
        args.no_test = True
        args.storeresults = False
    if args.training_type in ['ddp']:
        args.pre_post_epochs = args.epochs
        args.partition = True
    if args.model == 'small':
        args.dataset = 'mnist'
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    if args.training_type in [
            'ddp',
            'lSGD',
            'alSGD',
            'mnddp',
            'mnlSGD',
    ]:
        args.num_processes = 1
    """ The batch sizes are reduced by the batch size multiple """
    args.train_bs = args.train_processing_bs * args.bs_multiple
    args.test_bs = args.test_processing_bs * args.test_bs_multiple


def prepare_args(args):
    filter_args(args)
    args.host = platform.uname()[1].replace('.', '')
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if args.storeresults:
        prepare_dir(args)
        prepare_run_files(args)


def add_sys_args(parser):
    sys_parser = parser.add_argument_group(title='Directory etc. arguments')
    sys_parser.add_argument('--data-dir', default='~/data')
    sys_parser.add_argument('--imagenet-dir', default='~/imagenet_data')
    sys_parser.add_argument('--host', default='gpu118', help='Host name')


def add_dist_args(parser):
    dist_parser = parser.add_argument_group(title='dist arguments')
    dist_parser.add_argument('--dist-backend',
                             default='nccl',
                             help='dist mpi backend',
                             choices=['nccl', 'gloo', 'mpi'])
    dist_parser.add_argument(
        '--dist-url',
        default='tcp://127.0.0.1:21456',
        help='ip address and port of the machine running dist')
    dist_parser.add_argument('--numnodes',
                             type=int,
                             default=2,
                             help='number of nodes running dist')
    dist_parser.add_argument(
        '--masters_share',
        type=int,
        default=4,
        help='kth portion of dataset to be processed by the master node')


def add_lr_args(parser):
    lr_parser = parser.add_argument_group(title='LR args')
    lr_parser.add_argument('--scheduler-type',
                           default='const',
                           help='Scheduler type',
                           choices=['mstep', 'const', 'cosine'])
    lr_parser.add_argument('--gamma',
                           default=0.1,
                           type=float,
                           help='learning rate decay')
    lr_parser.add_argument('--eta', default=0.001, type=float, help='LARS eta')
    lr_parser.add_argument('--multiplier',
                           default=4,
                           type=int,
                           help='warm up multiplier')
    lr_parser.add_argument('--lrdecaystep',
                           default=30,
                           type=int,
                           help='learning rate decaystep')
    lr_parser.add_argument('--lrmilestone',
                           nargs='+',
                           type=int,
                           default=[150, 225],
                           help='learning rate decaystep')
    lr_parser.add_argument('--warm_up_epochs',
                           default=0,
                           type=int,
                           help='learning rate decaystep')
    lr_parser.add_argument('--prepassmepochs',
                           default=10,
                           type=int,
                           help='learning rate decaystep')
    lr_parser.add_argument('--baseline_lr',
                           default=0.1,
                           type=float,
                           help='baseline learning rate')


def add_averaging_args(parser):
    averaging_parser = parser.add_argument_group(title='averaging args')
    averaging_parser.add_argument('--averaging_freq',
                                  type=int,
                                  default=4,
                                  help='averaging interval in seconds')


def add_passm_args(parser):
    passm_parser = parser.add_argument_group(title='passm arguments')
    passm_parser.add_argument(
        '--pre_passm_epochs',
        type=int,
        default=100,
        metavar='N',
        help='The number of epochs in pre PASSM phase for PostPASSM')
    passm_parser.add_argument('--assmswitchepochs',
                              nargs='+',
                              type=int,
                              default=[50, 150, 180, 225, 250],
                              help='learning rate decaystep')
    passm_parser.add_argument('--no_assms_pre_passm',
                              action='store_true',
                              default=False,
                              help='used for postPassM training')

    passm_parser.add_argument('--dampen_for_passm',
                              action='store_true',
                              default=False,
                              help='used for PassM training')


def add_common_args(parser):
    common_parser = parser.add_argument_group(title='common arguments')
    common_parser.add_argument('--gpus',
                               nargs='+',
                               default=[1],
                               type=int,
                               help='GPU ranks')
    common_parser.add_argument('--lr',
                               default=0.1,
                               type=float,
                               help='learning rate')
    common_parser.add_argument('--momentum',
                               default=0.9,
                               type=float,
                               help='momentum')
    common_parser.add_argument('--nesterov',
                               default=False,
                               action='store_true',
                               help='nesterov momentum')
    common_parser.add_argument('--dampening',
                               default=0,
                               type=float,
                               help='momentum dampening')
    common_parser.add_argument('--weight-decay',
                               default=5e-4,
                               type=float,
                               help='L2 weight decay')
    common_parser.add_argument('--beta',
                               default=0.5,
                               type=float,
                               help='randBeta probab')
    common_parser.add_argument(
        '--model',
        default='res20',
        help='Neural Netowk Model',
        choices=[
            'small', 'ldense', 'vgg11', 'vgg13', 'vgg19', 'res18', 'res20',
            'res34', 'res32', 'res50', 'efficientnetb0', 'efficientnetb1',
            'efficientnetb2', 'efficientnetb3', 'wnet34', 'wnet50', 'wnet22',
            'wnet28', 'wnet168', 'rnet18x2', 'rnet50x2', 'densenet121',
            'mobilenetv2', 'mobilenetv3l', 'mobilenetv3s', 'shufflenet',
            'resnext50', 'densenet169', 'squuezenet', 'nasnet'
        ])
    common_parser.add_argument(
        '--dataset',
        default='cifar10',
        help='Dataset type',
        choices=['cifar10', 'cifar100', 'mnist', 'imagenet', 'catdog'])
    common_parser.add_argument('--num-classes',
                               type=int,
                               default=1000,
                               metavar='NC',
                               help='Number of classes')
    common_parser.add_argument('--training-type',
                               default='ddp',
                               help='Training algorithms type',
                               choices=['MBSGD', 'PLSGD', 'LAPSGD', 'LPPSGD'])
    common_parser.add_argument('--train_processing_bs', type=int, default=256)
    common_parser.add_argument('--test_processing_bs', type=int, default=256)
    common_parser.add_argument('--bs_multiple', type=int, default=1)
    common_parser.add_argument('--test_bs_multiple', type=int, default=1)
    common_parser.add_argument('--epochs', type=int, default=1)
    common_parser.add_argument('--pre_post_epochs', type=int, default=0)
    common_parser.add_argument('--test-freq', type=int, default=1)
    common_parser.add_argument('--tb-interval', type=int, default=5)
    common_parser.add_argument('--num_tags', type=int, default=5)
    common_parser.add_argument('--seed',
                               type=int,
                               default=1,
                               metavar='S',
                               help='random seed (default: 1)')
    common_parser.add_argument(
        '--num-processes',
        type=int,
        default=2,
        metavar='N',
        help='how many training processes to use (default: 2)')
    common_parser.add_argument('--num-threads',
                               type=int,
                               default=2,
                               metavar='N',
                               help='how many threads to use (default: 2)')
    common_parser.add_argument(
        '--workers',
        default=0,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 0)')
    common_parser.add_argument('--partition',
                               action='store_true',
                               default=True,
                               help='used for sparsified training')
    common_parser.add_argument('--cuda',
                               action='store_true',
                               default=False,
                               help='enables CUDA training')
    common_parser.add_argument('--lars',
                               action='store_true',
                               default=False,
                               help='enables LARS training')
    common_parser.add_argument('--distributed',
                               action='store_true',
                               default=False,
                               help='enables training on multiple GPUs')
    common_parser.add_argument('--dataparallel',
                               action='store_true',
                               default=False,
                               help='enables CUDA training')
    common_parser.add_argument('--pm',
                               action='store_true',
                               default=False,
                               help='enables memory-pinning')
    common_parser.add_argument('--no-progressbar',
                               action='store_true',
                               default=False,
                               help='do not show progressbar')
    common_parser.add_argument('--storeresults',
                               action='store_true',
                               default=False,
                               help='enables storing the results')
    common_parser.add_argument('--debug',
                               action='store_true',
                               default=False,
                               help='debug')
    common_parser.add_argument('--clean-start',
                               action='store_true',
                               default=False,
                               help='do test')
    common_parser.add_argument('--resume',
                               action='store_true',
                               default=False,
                               help='resume from checkpoint')


def prepare_experiment():
    parser = argparse.ArgumentParser(description='CNN Training')
    add_common_args(parser)
    add_dist_args(parser)
    add_lr_args(parser)
    add_sys_args(parser)
    add_averaging_args(parser)
    add_passm_args(parser)
    args = parser.parse_args()

    prepare_args(args)
    return args
