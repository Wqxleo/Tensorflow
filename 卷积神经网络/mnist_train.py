import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播函数
import 卷积神经网络.mnist_inference as mnistinfer
import numpy as np

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 使用指数衰减法设置学习率时可以将学习率设置的大一点
                          # 这样可以快速得到一个比较优的解，然后随着迭代的继续
                          # 逐步减小学习率，使得模型在后期更加稳定
LEARNING_RATE_DECAY = 0.99
REGULARATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "path/to/model2"
MODEL_NAME = "model2.ckpt"


def train(mnist):
    print("开始训练！")
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32, [BATCH_SIZE, mnistinfer.IMAGE_SIZE, mnistinfer.IMAGE_SIZE,
                     mnistinfer.NUM_CHANNELS], name='x-inpyt')
    y_ = tf.placeholder(
        tf.float32, [None, mnistinfer.NUM_LABELS], name='y-input')
    #正则化损失函数L2
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)
    # 直接使用mnist_inferense.py中定义的前向传播过程
    is_train = True
    y = mnistinfer.inference(x, is_train, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    # 交叉熵与softmax函数一起使用
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久类。

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("变量初始化！")
        tf.global_variables_initializer().run()

        # 在训练过程中不在测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnistinfer.IMAGE_SIZE,
                                          mnistinfer.IMAGE_SIZE,
                                          mnistinfer.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            # 每1000轮保存一次模型。
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    print("进入主函数！")
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print("准备训练！")
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
