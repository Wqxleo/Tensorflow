from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist.input_data
"""MNIST机器学习"""
"""下载并读取数据源"""
mnist = read_data_sets("MNIST_data/", one_hot=True)
"""x是784维占位符，基础图像28x28=784"""
x = tf.placeholder(tf.float32,[None,784])
"""W表示证据值向量，因为总数据量为0～9，每一维对应不同的数字，因此为10"""
W = tf.Variable(tf.zeros([784,10]))
"""b同理，为10"""
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
"""初始化交叉熵的值"""
y_ = tf.placeholder("float",[None,10])
"""计算交叉熵"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
"""梯度下降算法"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
"""开始训练模型"""
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
"""评估模型"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
"""正确率"""
k = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(k)