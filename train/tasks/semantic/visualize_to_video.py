#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
import __init__ as booger

# Add the path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from common.laserscan import LaserScan, SemLaserScan
from common.laserscanvis import LaserScanVis

class LaserScanVisualizer:
    """Class that handles visualization and saving of LiDAR scans as PNG images"""
    
    def __init__(self, scan, scan_names, label_names, output_dir, 
                 semantics=True, instances=False):
        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.output_dir = output_dir
        self.semantics = semantics
        self.instances = instances
        
        # Create output directories
        self.depth_dir = os.path.join(output_dir, "depth")
        self.semantic_dir = os.path.join(output_dir, "semantic")
        os.makedirs(self.depth_dir, exist_ok=True)
        if self.semantics:
            os.makedirs(self.semantic_dir, exist_ok=True)
    
    def process_scan(self, scan_idx):
        """Process a single scan and save as PNG"""
        # Open data
        if scan_idx < len(self.scan_names):
            self.scan.open_scan(self.scan_names[scan_idx])
        else:
            print(f"Warning: Scan {scan_idx} not found, using empty scan")
            # Create an empty scan
            self.scan.points = np.zeros((0, 3), dtype=np.float32)
            self.scan.remissions = np.zeros((0, 1), dtype=np.float32)
            self.scan.proj_range = np.full((self.scan.proj_H, self.scan.proj_W), -1, dtype=np.float32)
            self.scan.unproj_range = np.zeros((0, 1), dtype=np.float32)
            
        if self.semantics and scan_idx < len(self.label_names):
            self.scan.open_label(self.label_names[scan_idx])
            self.scan.colorize()
        
        # Get depth image
        power = 16
        depth_data = np.copy(self.scan.proj_range)
        depth_data_vis = np.copy(depth_data)
        
        # Handle empty depth data
        if np.all(depth_data_vis <= 0):
            depth_img = np.zeros((self.scan.proj_H, self.scan.proj_W, 3), dtype=np.uint8)
        else:
            depth_data_vis[depth_data_vis > 0] = depth_data_vis[depth_data_vis > 0]**(1 / power)
            if np.any(depth_data_vis > 0):
                min_val = depth_data_vis[depth_data_vis > 0].min()
                depth_data_vis[depth_data_vis < 0] = min_val
                depth_data_vis = (depth_data_vis - min_val) / (depth_data_vis.max() - min_val)
            
            # Convert to RGB image (viridis colormap)
            depth_img = plt.cm.viridis(depth_data_vis)
            depth_img = (depth_img[:, :, :3] * 255).astype(np.uint8)
        
        # Save depth image
        depth_filename = os.path.join(self.depth_dir, f"scan_{scan_idx:06d}.png")
        cv2.imwrite(depth_filename, cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR))
        
        # Process semantic data if available
        sem_filename = None
        if self.semantics:
            # Check if semantic projection has any data
            sem_img = np.copy(self.scan.proj_sem_color)
            
            # Fix for black semantics - check if the image is all black
            if np.max(sem_img) < 0.01:  # Almost black
                print(f"Warning: Scan {scan_idx} has black semantic data. Applying fallback coloring.")
                # Apply fallback coloring based on semantic labels
                mask = self.scan.proj_idx >= 0
                for y, x in zip(*np.where(mask)):
                    idx = self.scan.proj_idx[y, x]
                    if idx < len(self.scan.sem_label):
                        label = self.scan.sem_label[idx]
                        sem_img[y, x] = self.scan.sem_color_lut[label]
            
            # Convert to RGB image
            sem_img_rgb = (sem_img[:, :, ::-1] * 255).astype(np.uint8)
            
            # Save semantic image
            sem_filename = os.path.join(self.semantic_dir, f"scan_{scan_idx:06d}.png")
            cv2.imwrite(sem_filename, sem_img_rgb)
            
        return depth_filename, sem_filename
    
    def process_all_scans(self):
        """Process all scans and save as PNG images"""
        depth_files = []
        semantic_files = []
        
        # Use the maximum length of scan_names or label_names
        max_scans = max(len(self.scan_names), len(self.label_names)) if self.label_names else len(self.scan_names)
        
        for i in tqdm(range(max_scans), desc="Processing scans"):
            depth_file, sem_file = self.process_scan(i)
            depth_files.append(depth_file)
            if sem_file:
                semantic_files.append(sem_file)
        
        return depth_files, semantic_files
    
    def create_video(self, image_files, output_video_path, fps=10):
        """Create a video from a list of image files"""
        if not image_files:
            print(f"No images to create video at {output_video_path}")
            return
            
        # Read the first image to get dimensions
        img = cv2.imread(image_files[0])
        height, width, _ = img.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Add each image to the video
        for img_file in tqdm(image_files, desc=f"Creating video {os.path.basename(output_video_path)}"):
            img = cv2.imread(img_file)
            video.write(img)
            
        # Release the video writer
        video.release()
        print(f"Video saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser("./visualize_to_video.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
             'Must point to directory containing the predictions in the proper format '
             ' (see readme)'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default="visualization",
        required=False,
        help='Output directory for images and videos. Defaults to %(default)s',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        required=False,
        help='Frames per second for output video. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
             ', but if you are not done inferring this is not the case, so this disables'
             ' that safety.'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions_only',
        dest='predictions_only',
        default=False,
        action='store_true',
        help='Use only prediction files without requiring velodyne data.'
             'Defaults to %(default)s',
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)
    print("Predictions", FLAGS.predictions)
    print("Output Directory", FLAGS.output_dir)
    print("FPS", FLAGS.fps)
    print("ignore_semantics", FLAGS.ignore_semantics)
    print("ignore_safety", FLAGS.ignore_safety)
    print("predictions_only", FLAGS.predictions_only)
    print("*" * 80)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))
    
    # Create output directory
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.sequence)
    os.makedirs(output_dir, exist_ok=True)

    # Check for velodyne data
    scan_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "velodyne")
    scan_names = []
    
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
        # populate the pointclouds
        scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn]
        scan_names.sort()
    elif not FLAGS.predictions_only:
        print("Sequence folder doesn't exist! Trying to use predictions only...")
        FLAGS.predictions_only = True

    # Check for label/prediction data
    label_names = None
    if not FLAGS.ignore_semantics:
        if FLAGS.predictions is not None:
            label_paths = os.path.join(FLAGS.predictions, "sequences",
                                      FLAGS.sequence, "predictions")
        else:
            label_paths = os.path.join(FLAGS.dataset, "sequences",
                                      FLAGS.sequence, "labels")
        
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
            # populate the pointclouds
            label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_paths)) for f in fn]
            label_names.sort()
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()

        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety and not FLAGS.predictions_only and len(scan_names) > 0:
            assert (len(label_names) == len(scan_names))

    # create a scan
    if FLAGS.ignore_semantics:
        scan = LaserScan(project=True)  # project all opened scans to spheric proj
    else:
        color_dict = CFG["color_map"]
        scan = SemLaserScan(color_dict, project=True)

    # Create visualizer and process scans
    visualizer = LaserScanVisualizer(
        scan=scan,
        scan_names=scan_names,
        label_names=label_names,
        output_dir=output_dir,
        semantics=not FLAGS.ignore_semantics,
        instances=False
    )
    
    # Process all scans and get file paths
    depth_files, semantic_files = visualizer.process_all_scans()
    
    # Create videos
    visualizer.create_video(
        depth_files, 
        os.path.join(output_dir, "depth_video.mp4"), 
        fps=FLAGS.fps
    )
    
    if semantic_files:
        visualizer.create_video(
            semantic_files, 
            os.path.join(output_dir, "semantic_video.mp4"), 
            fps=FLAGS.fps
        )
    
    print("Visualization complete!")

if __name__ == '__main__':
    main() 