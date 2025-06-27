#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import sys

# Thêm đường dẫn để import các module từ thư mục gốc
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import train.__init__ as booger

from train.tasks.semantic.modules.user import *
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
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
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


    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
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
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # Sửa đổi cấu hình để phù hợp với dữ liệu hiện có
    if 'split' in DATA:
        # Chỉ giữ lại các sequence 0-4 cho train và 8 cho valid
        DATA['split']['train'] = [seq for seq in DATA['split']['train'] if int(seq) <= 4]
        DATA['split']['test'] = []  # Bỏ qua test

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # Sửa đổi User class để bỏ qua việc kiểm tra test
    class CustomUser(User):
        def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split, uncertainty, mc=30):
            super().__init__(ARCH, DATA, datadir, logdir, modeldir, split, uncertainty, mc)
            
        def infer(self):
            cnn = []
            knn = []
            if self.split == None:
                self.infer_subset(loader=self.parser.get_train_set(),
                                to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
                # do valid set
                self.infer_subset(loader=self.parser.get_valid_set(),
                                to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
            elif self.split == 'valid':
                self.infer_subset(loader=self.parser.get_valid_set(),
                                to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
            elif self.split == 'train':
                self.infer_subset(loader=self.parser.get_train_set(),
                                to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
            
            if len(cnn) > 0:
                print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
                print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
                print("Total Frames:{}".format(len(cnn)))
            print("Finished Infering")
            return

    # create user and infer dataset
    user = CustomUser(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.split, FLAGS.uncertainty, FLAGS.monte_carlo)
    user.infer()