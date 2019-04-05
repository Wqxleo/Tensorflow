import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py和mnist_train.py中定义的常量和函数
import 神经网络最佳实践.mnist_inference as mn_inference
import 神经网络最佳实践.mnist_train as mn_train

#每10秒加载一次最新模型，并在测试数据上测试最新模型的正确率。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出格式
        x = tf.placeholder(tf.float32, [None, mn_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mn_inference.OUTPUT_NODE], name='y-input')
        #x是训练图片，y_是对应的标签
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        #直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值，
        #所以这里用于计算正则化损失的函数被设置为None。
        y = mn_inference.inference(x, None)

        #使用前向传播的结果计算正确率。如果需要对位置的样例进行分类，那么使用
        #tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        #tf.argmax(y, 1)返回最大值的索引号，在这里y和y_都是藏长度为10的向量
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        #准确率  tf.cast类型转换  tf.reduce_mean求均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #通过变量重命名的方式来加载模型， 这样在前向传播的过程中就不需要调用求滑动平均
        #的函数来获取平均值了。这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mn_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                #tf.train.getcheckpoint_state函数或通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mn_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型。
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #通过文件名得到模型保存时的迭代轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("NO checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()