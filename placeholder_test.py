import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

#定义placeholder作为存放数据的地方。这里的维度也不一定要定义。
#但如果维度是确定的，那么给出的维度可以降低出错的概率。
#x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
with sess.as_default():
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    #print(sess.run(y))

    #print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))