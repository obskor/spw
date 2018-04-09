import tensorflow as tf
import numpy as np

a = [[1, 2],
     [3, 4]]
b = [[5, 6],
     [7, 8]]

sess = tf.Session()

c = np.multiply(a, b)
print(c)

d = tf.ones([2,2]) - a

sess.run(tf.global_variables_initializer())

print(sess.run(d))