#!/bin/bash

set -x

LOG="logs/fcn_`date +%Y-%m-%d_%H-%M-%S`.txt"
exec &> >(tee -a "$LOG")

python pre_building_edges.py ../../data/buildings/aoi_2/trainval_aug.txt

#exit
gpuid=1
./solve.py ${gpuid}


