"""
@author:  wangquaxiu
@time:  2019/1/13 16:17
"""
import numpy as np

# # numpy测试
# list = [1,2,3,4]
# oneArray = np.array(list)
# print(list)     #[1, 2, 3, 4]  这是python的列表对象
# print(oneArray) #[1 2 3 4]     这是一个一维数组

X = [1,2]
state = [0.0, 0.0]
# 分开定义不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1,0.2],[0.3, 0.4]])
w_cell_input = np.asarray([0.5,0.6])
b_cell = np.asarray([0.1,-0.1])

# 定义用于输出全连接层的参数
w_output = np.asarray([[1.0],[2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
    # 计算循环体中的全连接层神经网络
    before_activation = np.dot(state, w_cell_state)+ X[i]*w_cell_input+b_cell
    state = np.tanh(before_activation)

    # 根据当前时刻状态计算最终输出
    final_output = np.dot(state, w_output) + b_output

    # 输出每个时刻的信息
    print("before activation: ",before_activation)
    print("state: ", state)
    print("output: ", final_output)