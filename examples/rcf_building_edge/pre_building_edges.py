#!/usr/bin/env python
# Filename:
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 May, 2017
"""

from __future__ import division
import numpy as np
import sys,os

HOME = os.path.expanduser('~')

sys.path.insert(0, HOME+'/codes/rcf')
sys.path.insert(0, HOME+'/codes/rcf/examples/rcf_building_edge')

import basic.basic as basic
import basic.io_function as io_function
from PIL import Image
import cv2

import RCF_singlescale

from optparse import OptionParser

class SampleClass(object):
    image = ''      # path of image
    edge_map=''     # path of edge map produced by network
    groudT = ''     # path of groud image
    grounE = ''     # path of buidling edge
    id = ''         # file ID
# list of SampleClass
train_data = []

def read_train_data(train_file,file_id=None):
    """
    read the file list
    :param test_file: file tontains the test file list
    :param file_id: id is need in caffe for output result, not need in pytorch_deeplab_resnet
    :return: True if succeful, False otherwise
    """
    if os.path.isfile(train_file) is False:
        basic.outputlogMessage('error: file not exist %s'%train_file)
        return False
    f_obj = open(train_file)
    f_lines = f_obj.readlines()
    f_obj.close()

    if file_id is not None:
        fid_obj = open(file_id)
        fid_lines = fid_obj.readlines()
        fid_obj.close()

        if len(f_lines) != len(fid_lines):
            basic.outputlogMessage('the number of lines in test_file and test_file_id is not the same')
            return False

        for i in range(0,len(f_lines)):
            temp = f_lines[i].split()
            if len(temp) < 1:
                continue
            sample = SampleClass()
            sample.image = temp[0]
            if len(temp) > 1:
                sample.groudT = temp[1]
            sample.id = fid_lines[i].strip()
            train_data.append(sample)
    else:
        for i in range(0, len(f_lines)):
            temp = f_lines[i].split()
            if len(temp) < 1:
                continue
            sample = SampleClass()
            sample.image = temp[0]
            if len(temp) > 1:
                sample.groudT = temp[1]
                train_data.append(sample)

    # prepare file for pytorch_deeplab_resnet
    if len(train_data)< 1:
        basic.outputlogMessage('error, not input train data ')
        return False

    # check all image file and ground true file
    for sample in train_data:
        # check image path
        image_basename = os.path.basename(sample.image)
        if os.path.isfile(sample.image) is False:
            basic.outputlogMessage('error, file not exist: %s'%sample.image)
            return False

        # check ground path
        # if len(sample.groudT)>0 and os.path.isfile(sample.groudT) is False:
        #     sample.groudT = os.path.basename(sample.groudT)
        if  os.path.isfile(sample.groudT) is False:
            basic.outputlogMessage('error, file not exist: %s' % sample.groudT)
            return False

        if len(sample.id)< 1:
            sample.id = os.path.splitext(image_basename)[0]

    basic.outputlogMessage('read train data completed, sample count %d'%len(train_data))
    return True


def convert_groudT_to_groudEdge():
    if len(train_data) < 1:
        basic.outputlogMessage('error, no input images')
        return False
    for i in range(0,len(train_data)):
        groundT = train_data[i].groudT
        img_edge = os.path.join(os.path.split(groundT)[0],train_data[i].id + '_edge.png')
        train_data[i].groundE = img_edge

        print('buildings to edge : %d / %d'%(i,len(train_data)))

        im = Image.open(groundT)
        in_ = np.array(im, dtype=np.uint8)
        in_[np.where(in_ == 1)] = 0  # ignore building, only keep boundary
        in_[np.where(in_ == 255)] = 254

        w, h = in_.shape
        save_edge = np.empty((w, h,3),dtype=np.uint8)
        save_edge[:, :, 1] = save_edge[:, :, 2] = save_edge[:,:,0] = in_
        cv2.imwrite(img_edge, save_edge)

    pass

def produce_edge_map(train_data):
    if len(train_data) < 1:
        basic.outputlogMessage('error, no input images')
        return False



def main(options, args):
    building_list = args[0]
    if io_function.is_file_exist(building_list) is False:
        return False

    if read_train_data(building_list) is False:
        return False

    if len(train_data) < 1:
        basic.outputlogMessage('error, no input images')
        return False

    # if convert_groudT_to_groudEdge() is False:
    #     exit(1)
    save_root= os.path.join(os.getcwd(),'edge_map')
    io_function.mkdir(save_root)
    input_list = [item.image for item in train_data]
    gpuid =0
    edgethr = 100
    if options.gpuid is not None:
        gpuid = options.gpuid
    if options.edgeThr is not None:
        edgethr = options.edgeThr
    edge_map_list =RCF_singlescale.produce_edge_map(input_list,save_root,gpuid=gpuid, edgeThr=edgethr)

    save_txt = os.path.join(os.path.split(building_list)[0], 'edge_map.txt')
    fw_obj = open(save_txt, 'w')
    for i in range(0,len(train_data)):
        fw_obj.writelines('%s %s\n' % (train_data[i].image, edge_map_list[i]))
    fw_obj.close()


if __name__=='__main__':
    usage = "usage: %prog [options] test_lis_file"
    parser = OptionParser(usage=usage, version="1.0 2017-5-20")

    parser.add_option("-p", "--gpuid", action="store", dest="gpuid",type='int',
                      help="the id of gpu want to process")

    parser.add_option("-t", "--edgeThr", action="store", dest="edgeThr",type='int',
                      help="Pixel value which is smaller than edgeThr will be set as 0, it means edge")

    if len(sys.argv) < 2:
        parser.print_help()
        exit(1)

    (options, args) = parser.parse_args()
    main(options, args)