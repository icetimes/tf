# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 14:14
# @Author  : huyulan
# @Email   : huyulan@boe.com.cn
# @File    : tf-stack.py
# @Software: PyCharm
# coding:utf-8
import tensorflow as tf



a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8, 9], [0, 1, 7]])
s = tf.constant([[7, 8, 9], [0, 1, 7]])
c = tf.stack([a, b, s], axis=0)
x = c
ts = tf.transpose(x, perm=[1, 0, 2])
x = tf.unstack(ts)

with tf.Session() as sess:
    print("***c***")
    print(sess.run(c))
    print("***ts***")
    print(sess.run(ts))
    print("***x***")
    print(sess.run(x))
