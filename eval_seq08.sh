#!/bin/sh

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

helpFunction()
{
   echo "Usage: ./eval_seq08.sh [-d dataset_path] [-p prediction_path] [-m model_path] [-g gpu_id] [-c monte_carlo_samples] [-u uncertainty]"
   echo "  -d : Path to dataset directory (required)"
   echo "  -p : Path to prediction output directory (default: ./prediction)"
   echo "  -m : Path to model directory (default: ./pretrained)"
   echo "  -g : GPU ID to use (default: 1, use empty string for CPU)"
   echo "  -c : Number of Monte Carlo samples (default: 30)"
   echo "  -u : Use uncertainty model (default: false)"
   exit 1
}

# Default values
d="/media/data/eyecode-vinh/kitti/dataset/"
p="./prediction"
m="./pretrained"
g="1"
c="30"
u="false"

while getopts "d:p:m:g:c:u:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      g ) g="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      u ) u="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

# Check if required parameters are provided
if [ -z "$d" ] || [ -z "$p" ] || [ -z "$m" ]
then
   echo "Some or all of the required parameters are empty";
   helpFunction
fi

# Convert paths to absolute paths
d=$(get_abs_filename "$d")
p=$(get_abs_filename "$p")
m=$(get_abs_filename "$m")

# Set GPU device
export CUDA_VISIBLE_DEVICES="$g"

# Copy custom configuration to model directory
cp custom_data_cfg.yaml "$m/data_cfg.yaml"

echo "----------"
echo "PARAMETERS:"
echo "Dataset path: $d"
echo "Prediction path: $p"
echo "Model path: $m"
echo "GPU ID: $g"
echo "Monte Carlo samples: $c"
echo "Uncertainty: $u"
echo "----------"

# Run inference only on sequence 08 using the specialized script
cd ./train/tasks/semantic/; ./infer_seq08.py -d "$d" -l "$p" -m "$m" -u "$u" -c "$c"
echo "Inference completed on sequence 08 (validation set)."
