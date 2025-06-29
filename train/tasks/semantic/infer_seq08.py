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
import sys
import time
import traceback

from tasks.semantic.modules.user import *
from tasks.semantic.dataset.kitti.parser import Parser as KittiParser
from tasks.semantic.dataset.kitti.parser import SemanticKitti
import torch
import numpy as np
from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.postproc.KNN import KNN

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

# Tạo lớp Parser tùy chỉnh không yêu cầu dữ liệu training
class Seq08Parser:
    def __init__(self, root, train_sequences, valid_sequences, test_sequences, labels, 
                color_map, learning_map, learning_map_inv, sensor, max_points, 
                batch_size, workers, gt=True, shuffle_train=False):
        
        # Lưu các tham số
        self.root = root
        self.valid_sequences = valid_sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        
        # Số lớp cho cross entropy
        self.nclasses = len(self.learning_map_inv)
        
        # Chỉ tạo valid dataset
        self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   drop_last=True)
        
        # Kiểm tra valid loader có dữ liệu
        if len(self.validloader) == 0:
            raise ValueError("Validation set is empty or not available!")
            
        self.validiter = iter(self.validloader)
    
    def get_train_batch(self):
        # Không sử dụng train batch
        return None
    
    def get_train_set(self):
        # Không sử dụng train set
        return None
        
    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        return None

    def get_test_set(self):
        return None

    def get_train_size(self):
        return 0

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return 0

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)

# Lớp User tùy chỉnh cho sequence 08
class Seq08User:
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, uncertainty, mc=30):
        # Lưu các tham số
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.uncertainty = uncertainty
        self.mc = mc
        
        # Thiết lập device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Inference using", self.device)
        
        # Đảm bảo chỉ có sequence 08 trong valid_sequences
        if "8" not in DATA["split"]["valid"]:
            DATA["split"]["valid"] = ["8"]
        
        # Tạo parser tùy chỉnh
        self.parser = Seq08Parser(root=self.datadir,
                              train_sequences=[],  # Không cần train sequences
                              valid_sequences=self.DATA["split"]["valid"],
                              test_sequences=[],
                              labels=self.DATA["labels"],
                              color_map=self.DATA["color_map"],
                              learning_map=self.DATA["learning_map"],
                              learning_map_inv=self.DATA["learning_map_inv"],
                              sensor=self.ARCH["dataset"]["sensor"],
                              max_points=self.ARCH["dataset"]["max_points"],
                              batch_size=1,
                              workers=self.ARCH["train"]["workers"],
                              gt=True,
                              shuffle_train=False)
        
        # Tạo model
        self.model = self.make_model()
        
        # Tạo evaluator
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                self.device, self.uncertainty)
        
        # Tải model đã train
        self.model_path = os.path.join(self.modeldir, "SalsaNext")
        print(f"Loading model from: {self.model_path}")
        self.load_model()
        
        # Đặt model ở chế độ evaluation
        self.model.eval()
        self.model.eval_dropout = True if self.uncertainty else False
        
        # GPU?
        self.gpu = False
        self.model_single = self.model
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            self.gpu = True
            self.model.cuda()
            
        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                        self.parser.get_n_classes())
    
    def make_model(self):
        # Tạo model dựa trên cấu hình
        if self.uncertainty:
            model = SalsaNextUncertainty(self.parser.get_n_classes())
        else:
            model = SalsaNext(self.parser.get_n_classes())
        
        # Hỗ trợ nhiều GPU
        model = nn.DataParallel(model)
        
        return model
    
    def load_model(self):
        # Tải trọng số đã train
        try:
            print(f"Checking if model file exists: {os.path.exists(self.model_path)}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print("Loading model weights...")
            w_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            
            print("Model loaded successfully. Keys in state_dict:")
            if 'state_dict' in w_dict:
                print(f"Found 'state_dict' key with {len(w_dict['state_dict'])} parameters")
                self.model.load_state_dict(w_dict['state_dict'], strict=True)
                print("Model weights loaded successfully!")
            else:
                print(f"Available keys in model file: {list(w_dict.keys())}")
                raise KeyError("'state_dict' key not found in model file")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Detailed traceback:")
            traceback.print_exc()
            sys.exit(1)
        
    def infer_subset(self, loader, to_orig_fn, cnn, knn):
        # switch to evaluate mode
        if not self.uncertainty:
            self.model.eval()
        
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            end = time.time()
            
            # Lấy tổng số frames
            total_frames = len(loader)
            print(f"Processing {total_frames} frames from sequence 08...")
            
            # Đếm số frame đã xử lý
            processed = 0
            
            # Hiển thị thời gian bắt đầu
            start_time = time.time()
            last_update_time = start_time
            update_interval = 10  # Cập nhật mỗi 10 frames
            
            # Xử lý từng batch
            for proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints in loader:
                # first cut to real size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]
                
                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()
                
                # compute output
                if self.uncertainty:
                    proj_output_r, log_var_r = self.model(proj_in)
                    for i in range(self.mc):
                        log_var, proj_output = self.model(proj_in)
                        log_var_r = torch.cat((log_var, log_var_r))
                        proj_output_r = torch.cat((proj_output, proj_output_r))
                    
                    proj_output2, log_var2 = self.model(proj_in)
                    proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
                    log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
                    
                    # get predictions
                    proj_argmax = proj_output[0].argmax(dim=0)
                else:
                    proj_output = self.model(proj_in)
                    proj_argmax = proj_output[0].argmax(dim=0)
                
                # timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res_time = time.time() - end
                end = time.time()
                cnn.append(res_time)
                
                # knn postprocessing
                if self.post:
                    knn_start = time.time()
                    unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
                    knn_time = time.time() - knn_start
                    knn.append(knn_time)
                else:
                    unproj_argmax = proj_argmax[p_y, p_x]
                
                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)
                
                # map to original label
                pred_np = to_orig_fn(pred_np)
                
                # save scan
                path = os.path.join(self.logdir, "sequences", path_seq, "predictions", path_name)
                pred_np.tofile(path)
                
                # Cập nhật số frame đã xử lý
                processed += 1
                
                # Hiển thị tiến trình sau mỗi khoảng thời gian
                current_time = time.time()
                if processed % update_interval == 0 or processed == total_frames:
                    elapsed = current_time - start_time
                    frames_per_sec = processed / elapsed if elapsed > 0 else 0
                    eta = (total_frames - processed) / frames_per_sec if frames_per_sec > 0 else 0
                    
                    # Định dạng thời gian
                    eta_min = int(eta // 60)
                    eta_sec = int(eta % 60)
                    
                    progress = processed / total_frames * 100
                    print(f"Progress: {processed}/{total_frames} frames ({progress:.1f}%) | Speed: {frames_per_sec:.2f} fps | ETA: {eta_min}m {eta_sec}s")
            
            # Hiển thị thông tin kết thúc
            total_time = time.time() - start_time
            print(f"Processing completed! Total time: {total_time:.2f}s")
    
    def infer(self):
        cnn = []
        knn = []
        # Chỉ xử lý validation set (sequence 08)
        print("Inferring sequence 08 only...")
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        
        if len(cnn) > 0:
            print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
            print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
            print("Total Frames:{}".format(len(cnn)))
        print("Finished Inferring Sequence 08")
        return

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

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering sequence 08 only")
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

    # modify data config to only include sequence 08
    DATA["split"]["train"] = []
    DATA["split"]["test"] = []
    if "8" not in DATA["split"]["valid"]:
        DATA["split"]["valid"] = ["8"]

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        
        # only create directory for sequence 08
        seq = "08"
        print("Creating directory for sequence", seq)
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

    # create user and infer dataset
    user = Seq08User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.uncertainty, FLAGS.monte_carlo)
    user.infer() 