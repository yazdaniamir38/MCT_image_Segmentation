import time
import os
import math
import sys
#sys.path.append('C:/Users/auy200/venv/Lib/site-packages')
#import scipy.io as sci
import numpy as np
import tensorflow as tf
import cv2 as cvt
import h5py
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes
BATCH_SIZE = 16
EPOCH_SIZE = 10
MODEL_PATH = './model_weights_1/model_initial.ckpt'
MODEL_RESTORE = 0
#ORIENTATIONS = 12
#NUMBER_BPAT = 64
#NUMBER_NBPAT = 36
DATA_PATH= './256_3classes_3channels.h5'
#bpattern_Path='./64_extractedb2.h5'
#nbpattern_Path='./64_extractednb2.h5'
#NOISE_PATH = './64_Noise_DRIVE.h5'
DATA_NAMES = ['INPUT', 'TARGET']
h5fr = h5py.File(DATA_PATH, 'r')
s=np.arange(29833)
np.random.shuffle(np.arange(29833))
Input = h5fr[DATA_NAMES[0]][s,:,:,:]
Target = h5fr[DATA_NAMES[1]][s,:,:,:]
Weights = [np.count_nonzero(Target[:,:,:,0]), np.count_nonzero(Target[:,:,:,1]),np.count_nonzero(Target[:,:,:,2])]
#h5r_bpat = h5py.File(bpattern_Path, 'r')
#h5r_nbpat= h5py.File(nbpattern_Path, 'r')
#print(np.sum(Weights))
#bpatsize=bpat.shape
#nbpatsize=nbpat.shape
##DATA_SIZE = h5fr[DATA_NAMES[0]].shape
#h5fr_n = h5py.File(NOISE_PATH, 'r')

TRAINING_SIZE = 29833
TOTAL_BATCH_NUM = math.trunc(TRAINING_SIZE / BATCH_SIZE)
DISPLAY_INTERVAL = 10
PATCH_PATH = './log/'
dropout = 0.9
EPSILON = .001
TRAINING_RATE = .0001
#regularizing parameters
DIRECTION_FILTERS_SIZE = 22
NOISY_FILTER_SIZE = 64
NUMBER_NOISY = 100
import itertools as it

def xavierInitial(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# functions for regularized deep network
#generating patterns
def gaussian_fn(theta, xAxis, yAxis):
    xmin = -DIRECTION_FILTERS_SIZE/2
    xmax = DIRECTION_FILTERS_SIZE/2 - 1
    ymin = -DIRECTION_FILTERS_SIZE/2
    ymax = DIRECTION_FILTERS_SIZE/2 - 1
    (x, y) = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin,ymax+1))
    a1 = 1
    c1 = xAxis
    c2 = yAxis
    xNew = x*np.cos(theta) + y*np.sin(theta)
    yNew = -x*np.sin(theta) + y*np.cos(theta)
    f = a1* np.exp(-(((xNew **2)/ c1 **2) + ((yNew **2)/ c2 **2)))
    f = (f - f.min())/(f.max() - f.min())
    return f

def generate_noisyImage(input_image):
    for i in range(0,NUMBER_NOISY):
        temp = np.random.normal(0,(i + 1)*.05,[NOISY_FILTER_SIZE,NOISY_FILTER_SIZE])
        temp = (temp - temp.min())/(temp.max() - temp.min())
        input_image[:, :, 0, i] = temp
    return input_image



def regu_direction(filters, directions):
    loss = 0
    temp_directions = tf.transpose(directions, perm=[3,0,1,2])
    for angle in range(0, ORIENTATIONS):
        d =  temp_directions[angle,:,:,0]
        d = tf.reshape(d,[1,DIRECTION_FILTERS_SIZE,DIRECTION_FILTERS_SIZE,1])
        d = tf.cast(d, tf.float32)
        f = filters[:,:,0,angle]
        f = tf.reshape(f, [FILTER_SHAPE, FILTER_SHAPE, 1,1])
        f = tf.cast(f, tf.float32)
        out_same = tf.nn.conv2d(d, f, strides=[1, 1, 1, 1], padding='SAME')
        if angle < 6:
            stride = angle+6
        else:
            #???
            stride = angle-6
        d_ortho = temp_directions[stride,:,:,0]
        d_ortho = tf.reshape(d_ortho, [1, DIRECTION_FILTERS_SIZE, DIRECTION_FILTERS_SIZE, 1])
        d_ortho = tf.cast(d_ortho, tf.float32)
        out_ortho = tf.nn.conv2d(d_ortho, f, strides=[1, 1, 1, 1], padding='SAME')
        loss_same = tf.norm(out_same,ord=1)
        loss_ortho = tf.norm(out_ortho,ord=1)
        loss = loss + loss_ortho - loss_same
    loss = loss/(ORIENTATIONS)
    return loss
def regu_correlation(filters,bpat,nbpat ):
        loss = 0
        out_bone=tf.nn.conv2d(bpat, filters, strides=[1, 1, 1, 1], padding='SAME')
        out_nbone = tf.nn.conv2d(nbpat, filters, strides=[1, 1, 1, 1], padding='SAME')
        loss_bone = tf.norm(out_bone,axis=[1,2])
        loss_nbone = tf.norm(out_nbone,axis=[1,2])
        loss = loss + loss_bone - loss_nbonepo
        loss = loss/(bpatsize[0]+nbpatsize[0])
        return loss

def regu_noise(filters, noise):
    loss = 0
    temp_noise = noise
    #temp_noise = tf.transpose(noise, perm=[3,0,1,2])
    for i in range(0, NUMBER_NOISY):
        d = temp_noise[i, :, :, 0]
        d = tf.reshape(d, [1, NOISY_FILTER_SIZE, NOISY_FILTER_SIZE, 1])
        d = tf.cast(d, tf.float32)
        for angle in range(0, ORIENTATIONS):
            f = filters[:,:,0,angle]
            f = tf.reshape(f, [FILTER_SHAPE, FILTER_SHAPE, 1,1])
            f = tf.cast(f, tf.float32)
            out_same = tf.nn.conv2d(d, f, strides=[1, 1, 1, 1], padding='SAME')
            loss_same = tf.norm(out_same)
            loss = loss + loss_same
    loss = loss/(ORIENTATIONS*NUMBER_NOISY)
    return loss

## Gabor filter implementation
# FILTER_SHAPE = 11
# VAR_GABOR = 2.0/121.0
# filter_gabor = np.empty(shape=(FILTER_SHAPE, FILTER_SHAPE, 1, ORIENTATIONS))
# direction_filters = np.empty(shape=(DIRECTION_FILTERS_SIZE, DIRECTION_FILTERS_SIZE, 1, ORIENTATIONS))
# for angle in range(0, ORIENTATIONS):
#     params = {'ksize':(FILTER_SHAPE,FILTER_SHAPE), 'sigma':3.0, 'theta':angle*np.pi/ORIENTATIONS, 'lambd':15.0, 'gamma':.02}
#     filter_gabor[:,:,0,angle] = cvt.getGaborKernel(**params)
#     mean_gabor = np.mean(filter_gabor[:, :, 0, angle])
#     std_gabor = np.std(filter_gabor[:, :, 0, angle])
#     filter_gabor[:, :, 0, angle] = ((filter_gabor[:, :, 0, angle])/std_gabor)*np.sqrt(VAR_GABOR)
#     theta = angle*np.pi/ORIENTATIONS
#     direction_filters[:,:,0,angle] = gaussian_fn(theta, DIRECTION_FILTERS_SIZE/20,DIRECTION_FILTERS_SIZE/2)

#cvt.imshow('image', direction_filters[:,:,0,7])
#cvt.waitKey(0)
# params = {'ksize':(3,3), 'sigma':1.0, 'theta':0, 'lambd':15.0, 'gamma':.02}
# filter = cvt.getGaborKernel(**params)
# filter_gabor = tf.cast(filter_gabor,dtype=tf.float32)
stdv0=np.sqrt(2.0/(9.0))
stdv1 = np.sqrt(2.0/(9.0*12.0))
stdv2 = np.sqrt(2.0/(9.0*64.0))
stdv3 = np.sqrt(2.0/(9.0*128.0))
stdv4 = np.sqrt(2.0/(9.0*256.0))
stdv5 = np.sqrt(2.0/(9.0*512.0))
stdv_trans = np.sqrt(2.0/(9.0*16.0))
#w_conv_gabor = tf.get_variable('w_conv_gabor', initializer = filter_gabor, dtype=tf.float32)
w_conv0 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stdv0, name='w_conv0'))
# pool1 layers
w_conv1 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2, name='w_conv1'))
w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2, name='w_conv2'))
# pool2 layers
w_conv3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stdv2, name='w_conv3'))
w_conv4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=stdv3, name='w_conv4'))
# pool3 layers
w_conv5 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=stdv3, name='w_conv5'))
w_conv6 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=stdv4, name='w_conv6'))
w_conv7 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=stdv4, name='w_conv7'))
# 4th layer filters
w_conv8 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=stdv4, name='w_conv8'))
w_conv9 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=stdv5, name='w_conv9'))
w_conv10 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=stdv5, name='w_conv10'))
# output layer
w_conv_out = tf.Variable(tf.random_normal([3, 3, 64, 3], stddev=stdv2, name='w_conv_out'))

# uniform making layers
w_conv_layer3 = tf.Variable(tf.random_normal([3, 3, 512, 64], stddev=stdv5, name='w_conv_layer3'))
w_conv_layer2 = tf.Variable(tf.random_normal([3, 3, 256, 64], stddev=stdv4, name='w_conv_layer2'))
w_conv_layer1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=stdv3, name='w_conv_layer1'))
w_conv_layer0 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2, name='w_conv_layer0'))

# if transpose layer is used
w_transpose1 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2), name='w_transpose1')
w_transpose2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2), name='w_transpose2')
w_transpose3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv2), name='w_transpose3')

#b_conv0 = tf.Variable(tf.zeros([64]), name='b_conv0')
b_conv1 = tf.Variable(tf.zeros([64]), name='b_conv1')
b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')
b_conv3 = tf.Variable(tf.zeros([128]), name='b_conv3')
b_conv4 = tf.Variable(tf.zeros([128]), name='b_conv4')
b_conv5 = tf.Variable(tf.zeros([256]), name='b_conv5')
b_conv6 = tf.Variable(tf.zeros([256]), name='b_conv6')
b_conv7 = tf.Variable(tf.zeros([256]), name='b_conv7')
b_conv8 = tf.Variable(tf.zeros([512]), name='b_conv8')
b_conv9 = tf.Variable(tf.zeros([512]), name='b_conv9')
b_conv10 = tf.Variable(tf.zeros([512]), name='b_conv10')
b_conv_layer3 = tf.Variable(tf.zeros([64]), name='b_conv_layer3')
b_conv_layer2 = tf.Variable(tf.zeros([64]), name='b_conv_layer2')
b_conv_layer1 = tf.Variable(tf.zeros([64]), name='b_conv_layer1')
b_conv_layer0 = tf.Variable(tf.zeros([64]), name='b_conv_layer0')

b_conv_out = tf.Variable(tf.zeros([1]), name='b_conv_out')



# declaring inputs 0for network
input_cnn = tf.placeholder(tf.float32)
# bone= tf.placeholder(tf.float32)
# nbone = tf.placeholder(tf.float32)
weights=tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
#random_input = tf.plakeep_probceholder(tf.float32)
#tf.Variable(tf.random_normal([64, 40, 40, 1], stddev=1e-1), name='w_conv1')#
label_cnn = tf.placeholder(tf.float32)
input_shape = tf.shape(input_cnn)
input_shapeBy2 = input_shape/2
input_shapeBy4 = input_shape/4
input_shapeBy8 = input_shape/8
input_shapeBy2 = tf.cast(input_shapeBy2, tf.int32)
input_shapeBy4 = tf.cast(input_shapeBy4, tf.int32)
input_shapeBy8 = tf.cast(input_shapeBy8, tf.int32)


#h_bone = tf.nn.conv2d(bone, w_conv0, strides=[1, 1, 1, 1], padding='SAME')
#h_nbone = tf.nn.conv2d(nbone, w_conv0, strides=[1, 1, 1, 1], padding='SAME')

#loss_bone = tf.norm(h_bone)
#loss_bone = tf.reduce_mean(loss_bone)
#loss_nbone = tf.norm(h_nbone)
#loss_nbone = tf.reduce_mean(loss_nbone)


#rep_loss =loss_nbone -.5*loss_boneCannot specify both 'dim' and 'axis'

#implementing batch normalization
#input = tf.layers.batch_normalization(inputs=input_cnn, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends

# calculate regularizer losses
#loss_correlation = regu_correlation(w_conv0, bpat,nbpat)
#loss_noise = regu_noise(w_conv_gabor, random_input)

# Initial filtering with gabor for representation network
#h_gabor = tf.nn.conv2d(input_cnn, w_conv_gabor, strides=[1, 1, 1, 1], padding='SAME')
#implementing batch norkeep_probmalization
#h_gabor = tf.layers.batch_normalization(inputs=h_gabor, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends
# start implementing the regression network
# first layer implementation
h_conv0 = tf.nn.conv2d(input_cnn, w_conv0, strides=[1, 1, 1, 1], padding='SAME')
#implementing batch normalization
#h_conv0 = tf.layers.batch_normalization(inputs=input_cnn, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#h_conv0 = tf.nn.relu(h_conv0)
#Batch normalization ends
h_conv1 = tf.add(tf.nn.conv2d(h_conv0, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1)
h_conv1 = tf.layers.batch_normalization(inputs=h_conv1, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv1 = tf.nn.relu(h_conv1)
h_conv2 = tf.add(tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2)
#implementing batch normalization
h_conv2 = tf.layers.batch_normalization(inputs=h_conv2, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv2 = tf.nn.relu(h_conv2)
#Batch normalization ends
h_conv2 = tf.nn.dropout(h_conv2, keep_prob) 
input_2nd_Layer = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides= [1,2,2,1], padding = 'SAME')
# second layer implementationprint(loss)
h_conv3 = tf.add(tf.nn.conv2d(input_2nd_Layer, w_conv3, strides=[1, 1, 1, 1], padding='SAME'), b_conv3)
#implementing batch normalization
h_conv3 = tf.layers.batch_normalization(inputs=h_conv3, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv3 = tf.nn.relu(h_conv3)
#Batch normalization ends
h_conv4 = tf.add(tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME'), b_conv4)
#implementing batch normalization
h_conv4 = tf.layers.batch_normalization(inputs=h_conv4, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv4 = tf.nn.relu(h_conv4)
#Batch normalization ends
h_conv4 = tf.nn.dropout(h_conv4, keep_prob)
input_3rd_Layer = tf.nn.max_pool(h_conv4, ksize = [1,2,2,1], strides= [1,2,2,1], padding = 'SAME')
# third layer implementation
h_conv5 = tf.add(tf.nn.conv2d(input_3rd_Layer, w_conv5, strides=[1, 1, 1, 1], padding='SAME'), b_conv5)
#implementing batch normalizationloss_cnn_1 = tf.losses.mean_squared_error(label_cnn, h_output)
# loss_cnn_2 = tf.losses.absolute_difference(label_cnn, h_output)
# loss_cnn_2 = tf.reduce_mean(loss_cnn_2)
h_conv5 = tf.layers.batch_normalization(inputs=h_conv5, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv5 = tf.nn.relu(h_conv5)
#Batch normalization ends
h_conv6 = tf.add(tf.nn.conv2d(h_conv5, w_conv6, strides=[1, 1, 1, 1], padding='SAME'), b_conv6)
#implementing batch normalization
h_conv6 = tf.layers.batch_normalization(inputs=h_conv6, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv6 = tf.nn.relu(h_conv6)
#Batch normalization ends
h_conv7 = tf.add(tf.nn.conv2d(h_conv6, w_conv7, strides=[1, 1, 1, 1], padding='SAME'), b_conv7)
#implementing batch normalization
h_conv7 = tf.layers.batch_normalization(inputs=h_conv7, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv7 = tf.nn.relu(h_conv7)
#Batch normalization ends
h_conv7 = tf.nn.dropout(h_conv7, keep_prob)
input_4th_Layer = tf.nn.max_pool(h_conv7, ksize = [1,2,2,1], strides= [1,2,2,1], padding = 'SAME')
# fourth layer implementation
h_conv8 = tf.add(tf.nn.conv2d(input_4th_Layer, w_conv8, strides=[1, 1, 1, 1], padding='SAME'), b_conv8)
#implementing batch normalization
h_conv8 = tf.layers.batch_normalization(inputs=h_conv8, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv8 = tf.nn.relu(h_conv8)
#Batch normalization ends
h_conv9 = tf.add(tf.nn.conv2d(h_conv8, w_conv9, strides=[1, 1, 1, 1], padding='SAME'), b_conv9)
#implementing batch normalization
h_conv9 = tf.layers.batch_normalization(inputs=h_conv9, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv9 = tf.nn.relu(h_conv9)
#Batch normalization ends
h_conv10 = tf.add(tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME'), b_conv10)
#implementing batch normalization
h_conv10 = tf.layers.batch_normalization(inputs=h_conv10, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
h_conv10 = tf.nn.relu(h_conv10)
#Batch normalization ends
h_conv10 = tf.nn.dropout(h_conv10, keep_prob)
# upsampling and adding
h_conv_transpose_layer3 = tf.add(tf.nn.conv2d(h_conv10, w_conv_layer3, strides=[1, 1, 1, 1], padding='SAME'), b_conv_layer3)
h_conv_transpose_layer3 = tf.layers.batch_normalization(inputs=h_conv_transpose_layer3, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#implementing batch normalization
# h_conv_transpose_layer3 = tf.layeloss_cnn_1 = tf.losses.mean_squared_error(label_cnn, h_output)
# loss_cnn_2 = tf.losses.absolute_difference(label_cnn, h_output)
# loss_cnn_2 = tf.reduce_mean(loss_cnn_2)rs.batch_normalization(inputs=h_conv_transpose_layer3, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends
h_conv_transpose_layer2 = tf.add(tf.nn.conv2d(h_conv7, w_conv_layer2, strides=[1, 1, 1, 1], padding='SAME'), b_conv_layer2)
#implementing batch normalization
h_conv_transpose_layer2 = tf.layers.batch_normalization(inputs=h_conv_transpose_layer2, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends
h_conv_transpose_layer1 = tf.add(tf.nn.conv2d(h_conv4, w_conv_layer1, strides=[1, 1, 1, 1], padding='SAME'), b_conv_layer1)
#implementing batch normalization
h_conv_transpose_layer1 = tf.layers.batch_normalization(inputs=h_conv_transpose_layer1, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends
h_conv_transpose_layer0 = tf.add(tf.nn.conv2d(h_conv2, w_conv_layer0, strides=[1, 1, 1, 1], padding='SAME'), b_conv_layer0)
#implementing batch normalization
h_conv_transpose_layer0 = tf.layers.batch_normalization(inputs=h_conv_transpose_layer0, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends

input_upsample_layer2 = tf.nn.conv2d_transpose(h_conv_transpose_layer3, w_transpose1, output_shape=[BATCH_SIZE, input_shapeBy4[1],input_shapeBy4[2],64], strides=[1,2,2,1], padding='SAME')
#implementing batch normalization
#batch_mean, batch_var = tf.nn.moments(input_upsample_layer2,[0,1,2])
#scale15 = tf.Variable(tf.ones([64]), name='scale15')
#offset15 = tf.Variable(tf.zeros([64]), name='offset15')
#input_upsample_layer2  = tf.nn.batch_normalization(input_upsample_layer2,batch_mean,batch_var,offset15,scale15,EPSILON)
#Batch normalization ends
input_upsample_layer2 = input_upsample_layer2 + h_conv_transpose_layer2
#implementing batch normalization
input_upsample_layer2 = tf.layers.batch_normalization(inputs=input_upsample_layer2, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends
input_upsample_layer1 = tf.nn.conv2d_transpose(input_upsample_layer2, w_transpose2, output_shape=[BATCH_SIZE, input_shapeBy2[1],input_shapeBy2[2],64], strides=[1,2,2,1], padding='SAME')
input_upsample_layer1 = input_upsample_layer1 + h_conv_transpose_layer1
#implementing batch normalization
input_upsample_layer1 = tf.layers.batch_normalization(inputs=input_upsample_layer1, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends

input_upsample_layer0 = tf.nn.conv2d_transpose(input_upsample_layer1, w_transpose3, output_shape=[BATCH_SIZE, input_shape[1],input_shape[2],64], strides=[1,2,2,1], padding='SAME')
input_upsample_layer0 = input_upsample_layer0 + h_conv_transpose_layer0
#implementing batch normalization
input_upsample_layer0 = tf.layers.batch_normalization(inputs=input_upsample_layer0, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
#Batch normalization ends

# calculating output
h_output = tf.nn.conv2d(input_upsample_layer0, w_conv_out, strides=[1, 1, 1, 1], padding='SAME') + b_conv_out
#h_output = tf.nn.dropout(h_output, keep_prob)
# calculating the loss
# loss_cnn_1 = tf.losses.mean_squared_error(label_cnn, h_output)
# loss_cnn_2 = tf.losses.absolute_difference(label_cnn, h_output)
# loss_cnn_2 = tf.reduce_mean(loss_cnn_2)
loss=tf.nn.softmax_cross_entropy_with_logits(labels=label_cnn,logits=h_output)
if weights[0,2]!=0:
    wght=tf.reduce_sum(tf.multiply(label_cnn,1.955135*10**9*tf.reciprocal(weights)),axis=3)
    loss_cnn=tf.reduce_mean(tf.multiply(wght,loss))
else:
    loss_cnn=tf.reduce_mean(loss)
# loss_cnn = loss_cnn_1 + .1*loss_cnn_2 #+ 1e-9*rep_loss #+ 1e-5*loss_noise
#loss_cnn = loss_cnn_1 + .1*loss_cnn_2 + 1e-7*loss_direction

# training the network
# train_cnn = tf.train.MomentumOptimizer(.05, .9).minimize(loss_cnn)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_cnn = tf.train.AdamOptimizer(TRAINING_RATE).minimize(loss_cnn)
#train_cnn = tf.train.AdamOptimizer(.0005).minimize(loss_cnn)
accuracy = loss_cnn #tf.losses.mean_squared_error(label_cnn, h_output)
output_cnn = h_output
# feeding the input and labelsloss_cnn_1 = tf.losses.mean_squared_error(label_cnn, h_output)
# loss_cnn_2 = tf.losses.absolute_difference(label_cnn, h_output)
# loss_cnn_2 = tf.reduce_mean(loss_cnn_2)
#print(sess.run(temp_label))
saver = tf.train.Saver(tf.global_variables())
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input', input_cnn)
tf.summary.image('gt', label_cnn)
tf.summary.image('label', output_cnn)
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if MODEL_RESTORE == 1:
       saver.restore(sess, MODEL_PATH)
   # saver.restore(sess, MODEL_PATH)
    train_writer = tf.summary.FileWriter(PATCH_PATH, graph=tf.get_default_graph())
    step = 1
    #noise_data = h5fr_n[DATA_NAMES[0]][:,:,:,:]
    #bpat = h5r_bpat['INPUT'][:, :, :, :]
    #nbpat = h5r_bpat['INPUT'][:, :, :, :]
    for epoch in range(0, EPOCH_SIZE+1):
        for batch_index in range(0, TOTAL_BATCH_NUM):
            start_index = batch_index * BATCH_SIZE
            end_index = (batch_index + 1) * BATCH_SIZE
            input_data = Input[start_index:end_index, :, :, :]
            label_data = Target[start_index:end_index, :, :, :]
#            Weights=[[np.count_nonzero(label_data[:,:,:,0])],[np.count_nonzero(label_data[:,:,:,1])],[np.count_nonzero(label_data[:,:,:,2])]]

            # noisy_image = np.empty(shape=(NOISY_FILTER_SIZE, NOISY_FILTER_SIZE, 1, NUMBER_NOISY))
            # noisy_image = generate_noisyImage(noisy_image)
            # feed_dict = {input_cnn:input_data, label_cnn:label_data, bone:bpat, nbone:nbpat, keep_prob:dropout, is_training:True}
            feed_dict = {input_cnn: input_data, label_cnn: label_data, keep_prob: dropout,weights:np.reshape(Weights,[1,3]),is_training: True}
            # _,summary,accu, rloss = sess.run([train_cnn,merged,accuracy, rep_loss], feed_dict = feed_dict)
            kir, summary, accu = sess.run([train_cnn, merged, accuracy], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            step = step + 1
            if step % int(DISPLAY_INTERVAL) == 0:
                # print("\n[epoch %2.4f step %d] loss %.4f \t accuracy %.4f" % (epoch, step, loss, accu))
                # print("\n[epoch %2.4f step %d]  \t accuracy %.4f \t rloss %.4f" %(epoch, step, accu, rloss))
                print("\n[epoch %2.4f step %d]  \t accuracy %.4f" % (epoch, step, accu))
        if epoch%5==0:
            saver.save(sess, './model_weights_1/model_initial.ckpt')

print(" [*] Load SUCCESS")
