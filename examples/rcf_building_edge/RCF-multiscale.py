
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

data_root = '../../data/HED-BSDS/'
with open(data_root+'test.lst') as f:
    test_lst = f.readlines()
    
test_lst = [x.strip() for x in test_lst]


# In[3]:

im_lst = []
for i in range(0, len(test_lst)):
    im = Image.open(data_root+test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)


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
net = caffe.Net('test.prototxt', 'rcf_pretrained_bsds.caffemodel', caffe.TEST)

save_root = os.path.join(data_root, 'test-fcn')
if not os.path.exists(save_root):
    os.mkdir(save_root)


# In[6]:

start_time = time.time()
for idx in range(0, len(test_lst)):
    in_ = im_lst[idx]
    print ('produce edge map: %d of %d '%(idx,len(test_lst)))  
    
    scale = [0.5, 1, 1.5]
    multi_fuse = np.zeros(in_.shape[0:2], np.float32)
    for k in range(0, len(scale)):
        im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
        im_ = im_.transpose((2, 0, 1))
        
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *im_.shape)
        net.blobs['data'].data[...] = im_
        # run net and take argmax for prediction
        net.forward()
        
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
        fuse = cv2.resize(fuse, (in_.shape[1], in_.shape[0]), interpolation=cv2.INTER_LINEAR)
        multi_fuse += fuse
        
    multi_fuse /= len(scale)
    multi_fuse = 255 * (1-multi_fuse)
    cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_fuse.png', multi_fuse)
diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))


# In[ ]:



