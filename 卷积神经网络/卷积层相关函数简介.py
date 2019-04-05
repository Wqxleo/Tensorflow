import tensorflow as tf


#通过tf.get_variable的方式创建过滤器的权重变量和偏置项变量。卷积层的参数个数只和
#过滤器的尺寸、深度以及当前层节点矩阵的深度有关，所以这里声明的参数变量
#是一个四维矩阵，前面两个维度代表了过滤器的尺寸，第三个维度代表当前层的深度，
# 第四个维度表示过滤器的深度
filter_weight = tf.get_variable(
    'weight', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

#和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个
# 不同的偏置项。本样例代码中16为过滤器的深度，也是神经网络中下一层节点矩阵的深度
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1))

#tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播算法。这个函数的第一个输入
#为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一
#维对应一个输入batch。比如在输入层，input[0,:,:,:]表示第一张图片，input[1,:,:,:]表示
#第二张图片，以此类推。tf.nn.conv2d第二个参数提供了卷积层的权重，第三个参数为不同维度
#上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求一定
#是1。因为卷积层的步长只对矩阵的长和宽有效。最后一个参数是填充（padding）的方法，
#Tensorflow中提供SAME和VALID两种选择。其中SAME表示添加全0填充，VALID表示不添加。
conv = tf.nn.conv2d(
    input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

#tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加
#法，因为矩阵上不通位置上的节点都需要加上同样的偏置项。如图6-13所示，虽然下一层
#神经网络的大小为2*2，但是偏置项只有一个数（因为深度为1），而2*2矩阵中的每一个值
#都需要加上这个偏置项。
bias = tf.nn.bias_add(conv, biases)
#将计算结果通过ReLU激活函数完成去线性化
actived_conv = tf.nn.relu(bias)

"""池化层"""
#tf.nn.max_pool实现了最大池化层的前向传播过程，他的参数和tf.nn.conv2d函数类似。
#ksize提供了过滤器的尺寸，strides提供了步长信息，padding提供了是否使用全0填充。
pool = tf.nn.max_pool(
    actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
