#!/bin/sh

# Start timestamp
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "=== Script started at: $start_time ==="

# Set default values
d="/media/data/eyecode-vinh/kitti/dataset/"
p="/home/eyecode-vinh/SalsaNext/prediction"
m="/home/eyecode-vinh/SalsaNext/pretrained"
s="valid"  # Explicitly set to valid for sequence 08
g="1"  # GPU device ID, leave empty for CPU
c="30"  # Monte Carlo iterations
u="false"  # Uncertainty flag

# Function to get absolute path
get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# Parse command line arguments
while getopts "d:p:m:s:n:c:u:g:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      s ) s="$OPTARG" ;;
      n ) n="$OPTARG" ;;
      g ) g="$OPTARG" ;;
      u ) u="$OPTARG" ;;
      c ) c="$OPTARG" ;;
   esac
done

# Convert paths to absolute
d=$(get_abs_filename "$d")
p=$(get_abs_filename "$p")
m=$(get_abs_filename "$m")

# Set CUDA device if specified
if [ ! -z "$g" ]; then
    export CUDA_VISIBLE_DEVICES="$g"
fi

# Create prediction directory if it doesn't exist
mkdir -p "$p"

echo "=== Configuration ==="
echo "Dataset path: $d"
echo "Prediction path: $p"
echo "Model path: $m"
echo "Split: $s"
echo "GPU: $g"
echo "Monte Carlo iterations: $c"
echo "Uncertainty: $u"
echo "===================="

# Record inference start time
infer_start=$(date +%s)
echo "=== Starting inference at $(date +"%Y-%m-%d %H:%M:%S") ==="

# Run inference
cd ./train/tasks/semantic/ && python3 infer.py -d "$d" -l "$p" -m "$m" -s "$s" -u "$u" -c "$c"

# Record inference end time
infer_end=$(date +%s)
infer_duration=$((infer_end - infer_start))
echo "=== Inference completed at $(date +"%Y-%m-%d %H:%M:%S") ==="
echo "=== Inference took $infer_duration seconds ==="

# Run evaluation
echo "Starting evaluation"
eval_start=$(date +%s)
echo "=== Starting evaluation at $(date +"%Y-%m-%d %H:%M:%S") ==="

python3 evaluate_iou.py -d "$d" -p "$p" --split "$s"

# Record evaluation end time
eval_end=$(date +%s)
eval_duration=$((eval_end - eval_start))
echo "=== Evaluation completed at $(date +"%Y-%m-%d %H:%M:%S") ==="
echo "=== Evaluation took $eval_duration seconds ==="

# Calculate total execution time
total_duration=$((eval_end - $(date -d "$start_time" +%s)))
echo "=== Total execution time: $total_duration seconds ==="

# Calculate total execution time
total_duration=$((eval_end - $(date -d "$start_time" +%s)))
echo "=== Total execution time: $total_duration seconds ==="
echo "=== Script completed at: $(date +"%Y-%m-%d %H:%M:%S") ==="