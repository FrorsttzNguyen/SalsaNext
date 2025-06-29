#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import numpy as np

from tasks.semantic.modules.user import *
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer_seq08.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
                datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )

    FLAGS, unparsed = parser.parse_known_args()

    # Hardcode split to valid for sequence 08
    FLAGS.split = 'valid'
    
    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering ONLY on sequence 08")
    print("----------\n")
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        # Override splits to ONLY use sequence 08 for both train and validation
        # This prevents the AssertionError while focusing only on sequence 08
        DATA['split'] = {'train': ['08'], 'valid': ['08'], 'test': []}
        print("Overriding splits to run on sequence 08 ONLY.")
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        # Create directory for sequence 08 only
        seq = '08'
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
        os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise


    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    try:
        # Create a modified User class that only processes sequence 08
        class Seq08User(User):
            def infer(self):
                # Only run inference on validation set (sequence 08)
                print("Running inference ONLY on sequence 08 (validation set)")
                cnn = []
                knn = []
                self.infer_subset(loader=self.parser.get_valid_set(),
                                to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
                print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
                print("Total Frames:{}".format(len(cnn)))
                print("Finished Infering on sequence 08")
                return

        # Use our custom User class
        user = Seq08User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.split, FLAGS.uncertainty, FLAGS.monte_carlo)
        user.infer()
        print("Inference on sequence 08 completed successfully!")
    except Exception as e:
        print("Error during inference:", e)
        raise 