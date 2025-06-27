#!/bin/sh

get_abs_filename() {
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

helpFunction()
{
   echo "TODO"
   exit 1
}

while getopts "d:a:l:n:c:p:u:" opt
do
   case "$opt" in
      d ) d="$OPTARG" ;;
      a ) a="$OPTARG" ;;
      l ) l="$OPTARG" ;;
      n ) n="$OPTARG" ;;
      c ) c="$OPTARG" ;;
      p ) p="$OPTARG" ;;
      u ) u="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$a" ] || [ -z "$d" ] || [ -z "$l" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi
if [ -z "$u" ]
then u='false'
fi
d=$(get_abs_filename "$d")
a=$(get_abs_filename "$a")
l=$(get_abs_filename "$l")
if [ -z "$p" ]
then
 p=""
else
  p=$(get_abs_filename "$p")
fi
export CUDA_VISIBLE_DEVICES="$c"
cd ./train/tasks/semantic;  ./train.py -d "$d"  -ac "$a" -l "$l" -n "$n" -p "$p" -u "$u"



#https://drive.google.com/file/d/1j7iUSEWweLd-ks9jIck4RXpBoG9wOXMH/view?usp=sharing --00.zip
#https://drive.google.com/file/d/11EjgJBeTn24s6PQ_S6912ZZrr7iU4-Ic/view?usp=sharing --01.zip
#https://drive.google.com/file/d/1Nx1B3WivmZZXDOUp_OjZGrOr-aJ8AKAp/view?usp=sharing --02.zip
#https://drive.google.com/file/d/1WMm4mxeREazUwsMa6JwryJ0inOQDHd3N/view?usp=sharing --03.zip
#https://drive.google.com/file/d/1rysCfB0OtjvJRbaal70qOZMXOiQko4y1/view?usp=sharing --04.zip
#https://drive.google.com/file/d/1WMS1meTPRv3PYXNzEwehJVWKONxCKWVB/view?usp=sharing --05.zip
#https://drive.google.com/file/d/1Y8eW-CByC382wuB2VIQktz2LIC1CZlr3/view?usp=sharing --label
#https://drive.google.com/file/d/1SsqrIJmU4OfSr0OQjqRuXomE__vlErG9/view?usp=sharing --08.zip

#/media/data/eyecode-vinh/kitti/dataset/

