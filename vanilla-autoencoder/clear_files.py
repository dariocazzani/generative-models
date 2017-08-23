#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Authors:    Dario Cazzani
"""
import sys, os
sys.path.append('../')
from config import set_config
import subprocess

if __name__ == '__main__':

    parser = set_config()
    (options, args) = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_dir = dir_path.split('/')[-1]
    subprocess.call(['rm', '-r', '{}'.format(os.path.join(options.MAIN_PATH, cur_dir))])
    subprocess.call(['rm', '-r', 'out'])
    os.path.join(options.MAIN_PATH, cur_dir)
