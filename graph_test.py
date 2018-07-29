import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
# result = a + b
# print(tf.get_default_graph())
# print(a.graph is tf.get_default_graph())

#生成计算图g1
g1 = tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量"v"，并设置初始值为0
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

#生成计算图g2
g2 = tf.Graph()
with g2.as_default():
    #在计算图g2中定义变量"v"，并设置初始值为1
    v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer)

#在计算图g1中读取变量v的取值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

#在计算图g2中读取变量v的值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))