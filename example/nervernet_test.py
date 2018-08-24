import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
with sess.as_default():
    #自动初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # sess.run(w1.initializer) #初始化w1
    # sess.run(w2.initializer)#初始化w2
    print(sess.run(y))


#实现了神经网络的前向传播过程

