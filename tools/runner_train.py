#-*- coding: utf-8 -*-
import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse

from runner import Runner


def args_parser(args=None):
    parser = argparse.ArgumentParser("Training process...")
    parser.add_argument('--cfg', help='Config file', required=True)
    parser.add_argument('--alloc_mem', '-a',  help='Pre allocating GPU memory', action='store_true')
    return parser.parse_args(args)


if __name__=='__main__':
    if len(sys.argv[1:]):
        args = sys.argv[1:]
    else:
        args = None
    
    args = args_parser(args)

    # Initialize runner from a config file
    runner = Runner.from_cfg(args.cfg)

    # Start training
    runner.train()
