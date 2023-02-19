# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:09:11 2018

@author: Ben Chen
"""

import tensorflow as tf
from chessboards_output import traindatas

import os
import numpy as np
# 加载chessboards_infernece.py中定义的常量和前向传播的函数
import new_chessboards_infernece

# 配置神经网络的参数
dataset_size = 8000                 # 训练数据的总数目
BATCH_SIZE = 20                     # batch的样本数目
pixels = 307200			            # 单张图片
LEARNING_RATE_BASE = 0.02           # 基础学习率
LEARNING_RATE_DECAY = 0.98          # 学习率衰减率
REGULARIZATION_RATE = 0.0005        # 正则化参数
TRAINING_STEPS = 20000             # 训练次数
MOVING_AVERAGE_DECAY = 0.99         # 滑动平均率
#模型保存的路径及文件名
MODEL_SAVE_PATH = "p5chessboards_model"     # 模型保存路径
MODEL_NAME = "p5chessboards_model"          # 模型名称

def train(train_data, train_label):    # 训练程序
# 定义输出为4维矩阵的placeholder
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, [
                None,
                new_chessboards_infernece.IMAGE_SIZE1,
                new_chessboards_infernece.IMAGE_SIZE2,
                new_chessboards_infernece.NUM_CHANNELS],name='x-input')
        y_ = tf.placeholder(tf.float32, [None, new_chessboards_infernece.OUTPUT_NODE], name='y-input')
    
# 定义正则化函数
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
# 直接使用mnist_infernece.py中定义的前向传播过程
        y = new_chessboards_infernece.inference(x, False, regularizer)
    
# 定义储存训练轮数的变量。 这个变量不需要计算滑动平均值，所以这里的变量为不可训练的
# 变量（trainable=False）。
#在训练神经网络中一般会将代表训练论数的变量指定为不可训练的参数
        global_step = tf.Variable(0, trainable=False)

# 定义损失函数、学习率、滑动平均操作以及训练过程。

# 定义滑动平均衰减率和训动论数的变量，初始化滑动平均类。滑动平均可以增加数据的鲁棒性
# 给定训练论数的变量可以加快训练早期变量的更新速度
        variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
# 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step）就不需要了。
# tf.trainable_variables()返回的是GraphKeys.TRAINABLE_VARIABLES中的元素。
# 这个集合的元素就是所有没有指定trainable=False的参数
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
# 定义代价函数yi
    with tf.device('/gpu:1'):
        a = np.ones((BATCH_SIZE, pixels))*0.5
        loss = -tf.reduce_sum(tf.where(tf.greater(y_, a), 
                                       (y_*tf.log(tf.clip_by_value(y, 1e-6, 1.0)))/(tf.reduce_sum(y_)*1),
                                       (tf.log(1-tf.clip_by_value(y, 0, 0.999999)))/((BATCH_SIZE*pixels-tf.reduce_sum(y_))*1)))
    #a = np.ones((BATCH_SIZE, 76800))*0.5
    #loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))*76751)
# 设置指数衰减的学习率
        learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                dataset_size/BATCH_SIZE, LEARNING_RATE_DECAY,
                staircase=True)
# 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
                                                      global_step=global_step)

# 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络的参数，
# 也需要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了
# tf.control_dependencies和tf.group两种机制。下面两行程序和
# train_op = tf.group(train_step, variables_averages_op)是等价的
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

# 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        listrr = []
        for i in range(TRAINING_STEPS):
            #每次选取batch个样本进行训练
            start = (i*BATCH_SIZE)%dataset_size
            end = min(start+BATCH_SIZE,dataset_size)
            xs = train_data[start:end] 
            ys = train_label[start:end]

            reshaped_xs = np.reshape(xs, (
                xs.shape[0],
                new_chessboards_infernece.IMAGE_SIZE1,
                new_chessboards_infernece.IMAGE_SIZE2,
                new_chessboards_infernece.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], 
                                           feed_dict={x: reshaped_xs, y_: ys})

            listrr.append(loss_value)
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            #print(np.sum(ys))
        np.savetxt('cost_value5.txt',listrr)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
# 主程序入口
def main(argv=None):
# 声明处理 MNIST数据集的类，这个类会在初始化时自动下载数据
    train_data, train_label = traindatas()
    train(train_data, train_label)

if __name__ == '__main__':
    main()
