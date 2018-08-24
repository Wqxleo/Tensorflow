import  tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
c = tf.constant([[1.0], [2.0]])
d = tf.constant([[3.0, 4.0]])
result1 = tf.add(a, b, name="add")
result2 = tf.matmul(d, c, name="matmul")
print(result1)
print(result2)

#运行结果是一个张量的结构
#Tensor("add:0", shape=(2,), dtype=float32)
#Tensor("matmul:0", shape=(1, 1), dtype=float32)
#包含三个属性：name（名字）、维度（shape）和类型（type）
#name不仅是张量的唯一标识符，同时也给出了张量是如何计算出来的