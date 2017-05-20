
# coding: utf-8

# In[1]:

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
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# In[2]:

data_root = '../../data/NYUD/'
with open(data_root+'hha-test.lst') as f:
    hha_test_lst = f.readlines()
with open(data_root+'image-test.lst') as f:
    image_test_lst = f.readlines()
    
hha_test_lst = [x.strip() for x in hha_test_lst]
image_test_lst = [x.strip() for x in image_test_lst]


# In[3]:

hha_im_lst = []
for i in range(0, len(hha_test_lst)):
    im = Image.open(data_root+hha_test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((109.92,88.24,127.42))
    hha_im_lst.append(in_)
    
image_im_lst = []
for i in range(0, len(image_test_lst)):
    im = Image.open(data_root+image_test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    image_im_lst.append(in_)


# In[4]:

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


# In[5]:

#remove the following two lines if testing with cpu
#caffe.set_mode_gpu()
#caffe.set_device(0)
# load net
net1 = caffe.Net('test.prototxt', 'rcf_pretrained_nyud_depth.caffemodel', caffe.TEST)
net2 = caffe.Net('test.prototxt', 'rcf_pretrained_nyud_image.caffemodel', caffe.TEST)

save_root = os.path.join(data_root, 'test-fcn')
if not os.path.exists(save_root):
    os.mkdir(save_root)


# In[6]:

start_time = time.time()
for idx in range(0, len(hha_test_lst)):
    print ('produce edge map: %d of %d'%(idx,len(hha_test_lst)))
    hha_in_ = hha_im_lst[idx]
    hha_in_ = hha_in_.transpose((2, 0, 1))
    
    image_in_ = image_im_lst[idx]
    image_in_ = image_in_.transpose((2, 0, 1))
    
    assert hha_in_.shape == image_in_.shape,             'The HHA feature image must have the equal sizes with the RGB image...'
    
    # shape for input (data blob is N x C x H x W), set data
    net1.blobs['data'].reshape(1, *hha_in_.shape)
    net1.blobs['data'].data[...] = hha_in_
    # run net and take argmax for prediction
    net1.forward()
    
    # shape for input (data blob is N x C x H x W), set data
    net2.blobs['data'].reshape(1, *image_in_.shape)
    net2.blobs['data'].data[...] = image_in_
    # run net and take argmax for prediction
    net2.forward()
    
    # get output results of HHA net
    #out11 = net1.blobs['sigmoid-dsn1'].data[0][0, :, :]
    #out12 = net1.blobs['sigmoid-dsn2'].data[0][0, :, :]
    #out13 = net1.blobs['sigmoid-dsn3'].data[0][0, :, :]
    #out14 = net1.blobs['sigmoid-dsn4'].data[0][0, :, :]
    #out15 = net1.blobs['sigmoid-dsn5'].data[0][0, :, :]
    fuse1 = net1.blobs['sigmoid-fuse'].data[0][0, :, :]
    
    # get output results of RGB net
    #out21 = net2.blobs['sigmoid-dsn1'].data[0][0, :, :]
    #out22 = net2.blobs['sigmoid-dsn2'].data[0][0, :, :]
    #out23 = net2.blobs['sigmoid-dsn3'].data[0][0, :, :]
    #out24 = net2.blobs['sigmoid-dsn4'].data[0][0, :, :]
    #out25 = net2.blobs['sigmoid-dsn5'].data[0][0, :, :]
    fuse2 = net2.blobs['sigmoid-fuse'].data[0][0, :, :]
    
    #fuse = (out12+out13+out14+out22+out23+out24)/6
    fuse = (fuse1+fuse2)/2

    fuse = 255 * (1-fuse)    
    cv2.imwrite(save_root + '/' + hha_test_lst[idx][8:-4] + '_fuse.png', fuse)
    
diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(hha_test_lst))


# In[ ]:



