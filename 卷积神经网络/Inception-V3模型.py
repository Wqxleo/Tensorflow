import tensorflow as tf
import os

# slim.arg_scop函数可以用于设置默认的参数取值。slim.arg_scop函数的第一个参数是
# 一个函数列表，在这个列表中的函数将使用默认的参数取值。比如通过下面的定义，
# 调用slim.conv2d(net, 320, [1, 1])函数时会自动加上stride=1和padding='SAME'的参数。
# 如果在函数调用时指定了stride，那么这里设置的默认值就不会在使用，通过这种方式，
# 可以进一步减少冗余的代码。
from tensorflow.contrib import slim

with slim.arg_scop([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                   stride=1, padding='SAME'):
    # .....
    # 此处省略了Inception-v3模型中其他的网络结构二直接实现最后一层Inception结构
    # 假设输入图片经过之前的神经网络前向传播的结果保存在变量net中。
    # net = 上一层的输出节点
    net = None
    #为一个Inception模块声明一个统一的变量命名空间
    with tf.variable_scope('Maxed_7c'):
        # 给Inception模块中的每一条路径声明一个命名空间。
        with tf.variable_scope('Branch_0'):
            # 实现一个过滤器边长为1，深度为320的卷积层。
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        # Inception模块中的第二条路径。这条计算路径上的结构本身也是一个Inception结构。
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            # tf.concat函数可以将多个矩阵拼接起来。tf.concat函数的第一个参数指定了
            # 拼接的维度，这里给出的“3“代表了矩阵是在这个维度上进行拼接。
            branch_1 = tf.concat(3, [slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                                     slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')])

        # Inception模块中的第三条路径。此计算路径也是一个Inception结构。
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b3x3')
            branch_2 = tf.concat(3, [slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0d_3x1'),
                                     slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')])

        # Inception模块中的第四条路径
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

        # 当前Inception模块的最后输出是由上面四个计算结果拼接的到的。
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
