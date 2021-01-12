import time
import sys
sys.path.append('C:/Users/auy200/venv/Lib/site-packages')
import os
import math
#import scipy.io as sci
import numpy as np
import tensorflow as tf
import glob
import cv2
import scipy
from scipy import special
import h5py
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes
import ntpath
EXTENSION1 = 0
EXTENSION2 = 0

TEST_PATH = './Test/'
#TEST_PATH = './Test/ADNI/bicubic/'
#MODEL_PATH = './record_BW_DNSP/model9/model.ckpt'
MODEL_PATH = './model_weights_1/model_initial.ckpt'
#TEST_SAVE_PATH = './Test/BW_DNSP/results9/'
TEST_SAVE_PATH = './output_weights_1/'
BATCH_SIZE = 1
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

ORIENTATIONS = 12
FILTER_SHAPE = 11
VAR_GABOR = 2.0 / 121.0
filter_gabor = np.empty(shape=(FILTER_SHAPE, FILTER_SHAPE, 1, ORIENTATIONS))
E0PSILON = .0005
dropout = 1.0
EXTENSION = 2

with tf.Session() as sess:
    for angle in range(0, ORIENTATIONS):
        params = {'ksize': (FILTER_SHAPE, FILTER_SHAPE), 'sigma': 3.0, 'theta': angle * np.pi / ORIENTATIONS,
                  'lambd': 15.0, 'gamma': .02}
        filter_gabor[:, :, 0, angle] = cv2.getGaborKernel(**params)
        mean_gabor = np.mean(filter_gabor[:, :, 0, angle])
        std_gabor = np.std(filter_gabor[:, :, 0, angle])
        filter_gabor[:, :, 0, angle] = ((filter_gabor[:, :, 0, angle]) / std_gabor) * np.sqrt(VAR_GABOR)

    # cvt.imshow('image', filter_gabor[:,:,0,6])
    # cvt.waitKey(0)
    # params = {'ksize':(3,3), 'sigma':1.0, 'theta':0, 'lambd':15.0, 'gamma':.02}
    # filter = cvt.getGaborKernel(**params)
    filter_gabor = tf.cast(filter_gabor, dtype=tf.float32)
    stdv1 = np.sqrt(2.0 / (9.0 * 12.0))
    stdv2 = np.sqrt(2.0 / (9.0 * 64.0))
    stdv3 = np.sqrt(2.0 / (9.0 * 128.0))
    stdv4 = np.sqrt(2.0 / (9.0 * 256.0))
    stdv5 = np.sqrt(2.0 / (9.0 * 512.0))
    stdv_trans = np.sqrt(2.0 / (9.0 * 16.0))
    #w_conv_gabor = tf.get_variable('w_conv_gabor', initializer=filter_gabor, dtype=tf.float32)

    # pool1 layers
    w_conv0 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stdv1, name='w_conv0'))
    w_conv1 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=stdv1, name='w_conv1'))
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

    # declaring inputs for network
    input_cnn = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    # tf.Variable(tf.random_normal([64, 40, 40, 1], stddev=1e-1), name='w_conv1')#
    label_cnn = tf.placeholder(tf.float32)
    input_shape = tf.shape(input_cnn)
    input_shapeBy2 = input_shape / 2
    input_shapeBy4 = input_shape / 4
    input_shapeBy2 = tf.cast(input_shapeBy2, tf.int32)
    input_shapeBy4 = tf.cast(input_shapeBy4, tf.int32)

    #implementing batch normalization
    #input = tf.layers.batch_normalization(inputs=input_cnn, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
    #Batch normalization ends
    
    # Initial filtering with gabor for representation network
    #h_gabor = tf.nn.conv2d(input_cnn, w_conv_gabor, strides=[1, 1, 1, 1], padding='SAME')
    #implementing batch normalization
    #h_gabor = tf.layers.batch_normalization(inputs=h_gabor, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
    #Batch normalization ends
    # start implementing the regression network
    # first layer implementation
    h_conv0 = tf.nn.conv2d(input_cnn, w_conv0, strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.add(tf.nn.conv2d(h_conv0, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1)
    #implementing batch normalization
    h_conv1 = tf.layers.batch_normalization(inputs=h_conv1, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
    h_conv1 = tf.nn.relu(h_conv1)
    #Batch normalization ends
    h_conv2 = tf.add(tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2)
    #implementing batch normalization
    h_conv2 = tf.layers.batch_normalization(inputs=h_conv2, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
    h_conv2 = tf.nn.relu(h_conv2)
    #Batch normalization ends
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob) 
    input_2nd_Layer = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides= [1,2,2,1], padding = 'SAME')
    # second layer implementation
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
    #implementing batch normalization
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
    #implementing batch normalization
    h_conv_transpose_layer3 = tf.layers.batch_normalization(inputs=h_conv_transpose_layer3, axis=-1, momentum=0.999,  epsilon=0.001, center=True, scale=True, training = is_training)
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
    # Model done................
    output_cnn = h_output
    # Loading the test input and the model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, MODEL_PATH)
    print(glob.glob(TEST_PATH + '*.png'))
    for testImgName in glob.glob(TEST_PATH + '*.png'):
        print('Test Image %s'% path_leaf(testImgName))
        testName = str(path_leaf(testImgName))
        #testName = testName[6:]
        
        testImg = cv2.imread(testImgName, 0)
        testImg_normalized = testImg/255.0
        (h,w) = testImg_normalized.shape[0:2]
        if h % 256 != 0:
            EXTENSION1 = 256 - (h%256)
        if w % 256 !=0:
            EXTENSION2 = 256 - (w%256)
        
        temp = np.zeros((h+EXTENSION1,w+EXTENSION2))
        print(temp.shape)
        temp[:h,:w] = testImg_normalized
        ## gabor testing ends
        test_input = np.array([temp])
        test_elem = np.rollaxis(test_input, 0,3)
        test_data = test_elem[np.newaxis, ...]
        output_image = np.zeros([h+EXTENSION1, w+EXTENSION2, 3])
        for i in range(int((h+EXTENSION1)/256)):
            for j in range(int((w+EXTENSION2)/256)):
                output_data = sess.run([output_cnn], feed_dict={input_cnn:test_data[:,256*i:256*(i+1),256*j:256*(j+1),:],keep_prob:dropout, is_training:False})
        # output_data=tf.nn.softmax(output_data)
        #     output_image=np.zeros([output_data[0].shape[1],output_data[0].shape[2],3])
        # output_image = output_data[0][0,:,:,:]
        # output_image[np.nonzero(output_data[0][0,:,:,0]),0] = output_image[:h,:w]
                ind=np.nonzero((np.argmax(special.softmax(output_data[0],axis=3),axis=3)==0)[0])
                output_image[ind[0]+256*i,ind[1]+256*j,0] = 1
                ind = np.nonzero((np.argmax(scipy.special.softmax(output_data[0], axis=3), axis=3) == 1)[0])
                output_image[ind[0] + 256 * i, ind[1] + 256 * j,1] = 1
                ind = np.nonzero((np.argmax(scipy.special.softmax(output_data[0], axis=3), axis=3) == 2)[0])
                output_image[ind[0] + 256 * i, ind[1] + 256 * j,2] = 1
        # output_image[(np.argmax(scipy.special.softmax(output_data[0], axis=3), axis=3) == 1)[0,:,:], 1] = 1
        # output_image[(np.argmax(scipy.special.softmax(output_data[0], axis=3), axis=3) == 2)[0,:,:], 2] = 1
        output_image=output_image[:h,:w]
        output_image = output_image*255
        EXTENSION1 = 0
        EXTENSION2 = 0

        # cv2.imshow('image',output_image)
        # cv2.waitKey(0)
        test_save_name = TEST_SAVE_PATH + testName
        cv2.imwrite(test_save_name, output_image)

    print('>>>Start shuffling Images:')

    # mat = sci.loadmat('blurr1.mat')
    # temp_input = np.array(mat['ylrNew'])
    # temp_input = np.reshape(temp_input, [1,10,10,1])
    # temp_input = np.true_divide(temp_input, 255.0)
    #
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, './record/model.ckpt')
    # # graph = tf.get_default_graph()
    # print("Model restored!")
    # accu = sess.run(loss_cnn, feed_dict={input_cnn: temp_input, label_cnn: temp_label})
    # print("Reload accuracy:{}".format(accu))


    #
    # feed_dict={input_cnn: temp_input, label_cnn: temp_label}
    # print(sess.run(w_conv1))
    # print(sess.run(w_conv1, feed_dict))
