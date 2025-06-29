#!/bin/sh

# Paths
dataset="/media/data/eyecode-vinh/kitti/dataset/"
model="/home/eyecode-vinh/SalsaNext/pretrained"
prediction="/home/eyecode-vinh/SalsaNext/prediction_seq08"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate salsanext

# Run inference on sequence 08
cd ./train/tasks/semantic/
./infer_seq08.py -d "$dataset" -m "$model" -l "$prediction"

echo "Inference completed on sequence 08." 