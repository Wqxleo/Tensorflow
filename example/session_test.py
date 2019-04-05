import  tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result1 = tf.add(a, b, name="add")
sess = tf.Session()
print(result1)
with sess.as_default():
    print(sess.run(result1))
    print(result1.eval())

#
#
# # print(result1.eval(session=sess))
#
# sess.close()