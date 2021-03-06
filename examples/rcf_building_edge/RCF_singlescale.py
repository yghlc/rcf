
# coding: utf-8

# In[8]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
#get_ipython().magic(u'matplotlib inline')
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2
import time

# Make sure that caffe is on the python path:
HOME = os.path.expanduser('~')
caffe_root = HOME+'/codes/rcf/'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# In[15]:

# Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size / 2

    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1, 5, i + 1)
        plt.imshow(1 - scale_lst[i], cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()

def produce_edge_map(input_list,save_root, edgeThr=100,gpuid=0,test_pro='test.prototxt',pre_trained='rcf_pretrained_bsds.caffemodel'):
    # files in input_list are absolute path

    test_lst = [x.strip() for x in input_list]

    im_lst = []
    edge_map_list = []
    for i in range(0, len(test_lst)):
        im = Image.open(test_lst[i])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        im_lst.append(in_)



    #remove the following two lines if testing with cpu
    if gpuid >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)
    # load net
    net = caffe.Net(test_pro, pre_trained, caffe.TEST)

    # save_root = os.path.join(data_root, 'test-fcn')
    # print save_root
    # if not os.path.exists(save_root):
    #     os.mkdir(save_root)


    # In[22]:

    start_time = time.time()
    for idx in range(0, len(test_lst)):
        in_ = im_lst[idx]
        in_ = in_.transpose((2, 0, 1))
        print ('produce edge map: %d of %d : %s'%(idx,len(test_lst),test_lst[idx]))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()

        # save results
        #out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
        #out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
        #out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
        #out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
        #out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
        #out1 = 255 * (1-out1)
        #out2 = 255 * (1-out2)
        #out3 = 255 * (1-out3)
        #out4 = 255 * (1-out4)
        #out5 = 255 * (1-out5)
        fuse = 255 * (1-fuse)
        #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out1.png', out1)
        #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out2.png', out2)
        #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out3.png', out3)
        #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out4.png', out4)
        #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out5.png', out5)

        fuse[fuse<=edgeThr] = 0
        fuse[fuse>edgeThr] = 255

        save_name = os.path.splitext(os.path.basename(test_lst[idx]))[0] + '_edge.png'
        save_path = os.path.join(save_root,save_name)
        edge_map_list.append(save_path)

        cv2.imwrite(save_path , fuse)

    diff_time = time.time() - start_time
    print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))

    return edge_map_list


if __name__=='__main__':
    pass



