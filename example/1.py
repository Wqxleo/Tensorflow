"""
@author:  wangquaxiu
@time:  2019/3/15 19:20
"""
import tensorflow as tf
a = tf.constant([1.0, 2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
# result = tf.add(a,b,name='add')
result = a+b
print(result)