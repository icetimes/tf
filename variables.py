# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 11:33
# @Author  : huyulan
# @Email   : huyulan@boe.com.cn
# @File    : testt.py
# @Software: PyCharm

import tensorflow as tf
with tf.variable_scope("foo"):
    v = tf.get_variable('v',1)
    t = 2*v

with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable('v',[1])

assert v1 is v
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(t))

