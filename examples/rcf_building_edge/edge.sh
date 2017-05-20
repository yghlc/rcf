#!/usr/bin/env bash

expr=~/experiment/caffe_deeplab/aoi_2
code_path=~/codes/rcf/examples/rcf_building_edge

file_list=${1}
gpuid=${2}

echo file_list: ${file_list} , GPU ID: ${gpuid}

cd ${code_path}
git pull
cd -

cp ${code_path}/RCF_singlescale.py .
cp ${code_path}/pre_building_edges.py .
cp ${code_path}/test.prototxt .
cp ${code_path}/rcf_pretrained_bsds.caffemodel .

python pre_building_edges.py ${file_list} --gpuid=${gpuid} --edgeThr=200

