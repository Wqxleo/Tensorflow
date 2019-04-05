"""
@author:  wangquaxiu
@time:  2019/1/14 9:26
"""

"""

       伪代码

"""


import tensorflow as tf
from tensorflow.python.ops import rnn_cell
# 定义一个LSTM结构。在Tensorflow中通过一句简单的命令就可以实现一个完整的LSTM结构。
# LSTM中使用的变量也会在函数中自动被声明

lstm_hidden_size = 10
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)
batch_size = 5
# 将LSTM中的状态初始化为全0数组。和其他神经网络类似，在优化神经网络时，每次也会
# 使用一个batch的训练样本。在以下代码中，batch_size给出了一个batch的大小。
# BasicLSTMCell类提供了zero_state的函数来生成全0的初始状态。

state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0
# 在8.1节中介绍过，虽然理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免
# 梯度消散的问题，会规定一个最大的序列长度，在以下代码中，用num_steps来表示这个长度。
for i in range(10):
    # 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。
    if i > 0:
        tf.get_variable_scope().reuse_variables()
        # 每一步处理时间序列中的一个时刻。将当前输入（current_input)和前一时刻状态
        # （state）传入定义的LSTM结构可以得代当前LSTM结构的输出lstm_output和更新后
        # 的状态state
        lstm_output, state = lstm(current_input, state)
        # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
        final_output = fully_connected(lstm_output)
        # 计算当前时刻输出的损失
        loss += calc_loss(final_output, excepted_output)
