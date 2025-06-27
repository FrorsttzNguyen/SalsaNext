#!/bin/sh

d="/media/data/eyecode-vinh/kitti/dataset/"
p="/home/eyecode-vinh/SalsaNext/prediction"
m="/home/eyecode-vinh/SalsaNext/pretrained"
s="valid"
g="1"  # Để trống để sử dụng CPU
c="30"

helpFunction()
{
   echo "Options not found"
   exit 1
}

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

while getopts "d:p:m:s:n:c:u:g" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      m ) m="$OPTARG" ;;
      s ) s="$OPTARG" ;;
      n ) n="$OPTARG"  ;;
      g ) g="$OPTARG" ;;
      u ) u="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$d" ] || [ -z "$p" ] || [ -z "$m" ]
then
   echo "Some or all of the options are empty";
   helpFunction
fi
if [ -z "$u" ]
then u='false'
fi
d=$(get_abs_filename "$d")
p=$(get_abs_filename "$p")
m=$(get_abs_filename "$m")
export CUDA_VISIBLE_DEVICES="$g"

# Sao chép file cấu hình tùy chỉnh vào thư mục mô hình
cp custom_data_cfg.yaml "$m/data_cfg.yaml"

# Chỉ chạy inference trên sequence 08 (tập validation)
cd ./train/tasks/semantic/; ./infer.py -d "$d" -l "$p" -m "$m" -n "$n" -s "valid" -u "$u" -c "$c"
echo "Inference completed on validation set (sequence 08)."