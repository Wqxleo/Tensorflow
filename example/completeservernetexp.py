# 总结：
# 训练神经网络的过程可以分为以下三个步骤：
# 1、定义神经网络的结构和前向传播的输出结果
# 2、定义损失函数以及选择反向传播优化的算法
# 3、生成会话（tf.Session）并且在训练数据上反复运行反向传播优化算法



import tensorflow as tf
#NumPy是一个科学计算的工具包，这里通过NumPy生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#在shape的一个维度上使用None可以方便使用不大的batch大小。在训练师要把数据
# 分成比较小的batch，但在测试时，可以一次性使用全部的数据。当数据及比较小时
# 这样比较方便测试，但数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

#定义神经网络向前传播的过程。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播算法。
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#Tensorflow经过使用Adam优化算法对损失函数中变量进行修改值
#默认修改tf.Variable类型的参数。
#也可以使用var_list参数来定义更新哪些参数
# 如：train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy，var_list=[w1, w2])


#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本
# 而其他为负样本。和Tensorflow游乐场中的表示法不大一样的地方是，
# 在这里用0来表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用
# # 0和1的表示方法
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

#创建一个回话来运行Tensorflow程序
with tf.Session() as sess:
    iniy_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(iniy_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        #通过选取的样本训练神经网络并更新参数

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            #每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g"%(i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
