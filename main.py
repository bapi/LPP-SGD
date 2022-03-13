"""This is the main function script"""

import time
import sys


def main():
    from utilities.args import prepare_experiment
    args = prepare_experiment()
    # print(args)
    start = time.perf_counter()
    if args.training_type == 'LPPSGD':
        from train.LPPSGD import run
        run(args)
    elif args.training_type == 'LAPSGD':
        from train.LAPSGD import run
        run(args)
    elif args.training_type == 'MBSGD':
        from train.MBSGD import run
        run(args)
    elif args.training_type == 'PLSGD':
        from train.PLSGD import run
        run(args)
    else:
        sys.exit("Wrong training-type!")
    finish = time.perf_counter()
    print('Total_time=' + str(' % d' % (finish - start)))


if __name__ == '__main__':
    main()
