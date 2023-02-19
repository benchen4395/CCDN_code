# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:22:38 2017

@author: Ben Chen
"""

# This is realized by CNN
import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 307200
OUTPUT_NODE = 307200
# 图像的尺寸、通道、图像标签数
IMAGE_SIZE1 = 480
IMAGE_SIZE2 = 640
NUM_CHANNELS = 1
NUM_LABELS = 307200

# 第一层卷积的尺寸和深度
CONV1_DEEP = 20
CONV1_SIZE = 9

# 第二层卷积的尺寸和深度
CONV2_DEEP = 20
CONV2_SIZE = 3

# 第二层卷积的尺寸和深度
CONV3_DEEP = 1
CONV3_SIZE = 3

# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数train, 用于区分训练过程和测试过程。
# 在这个程序中将用到dropout方法，dropout方法可以进一步提高模型可靠性并防止过拟合
# dropout方法只在训练中使用。
def inference(input_tensor, train, regularizer):
    
# 实现第一层卷积层的前向传播过程
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],  dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

# 实现第一层池化层的前向传播过程
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],
                               strides=[1,1,1,1],padding="SAME")

# 实现第二层卷积层的前向传播过程
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],  dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
# 实现第三层卷积层的前向传播过程
    with tf.variable_scope("layer4-conv3"):
        conv3_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV2_DEEP, CONV2_DEEP],  dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV2_DEEP], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(relu2, conv3_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
# 实现第四层卷积层的前向传播过程
    with tf.variable_scope("layer5-conv4"):
        conv4_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV2_DEEP, CONV2_DEEP],  dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV2_DEEP], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

# 实现第二层池化层的前向传播过程
    with tf.name_scope("layer6-pool5"): 
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], 
                               strides=[1, 1, 1, 1], padding='SAME')
        
# 实现第五层卷积层的前向传播过程
    with tf.variable_scope("layer7-conv5"):
        conv5_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],  dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV3_DEEP],  dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2, conv5_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
# 实现第六层卷积层的前向传播过程
    with tf.variable_scope("layer8-conv6"):
        conv6_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV3_DEEP, CONV3_DEEP],  dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [CONV3_DEEP],  dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_weights, 
                             strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

     
# pool2.get_shape()函数可以得到第六层的输出矩阵的维度，然后再将其转化成列表
    pool_shape = relu6.get_shape().as_list()

# 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长、宽以及深度的乘积。
# pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        
# 通过tf.reshape()函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(relu6, [-1, nodes])
        
# 返回第六层的输出
    return reshaped
