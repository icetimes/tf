# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 14:14
# @Author  : huyulan
# @Email   : huyulan518@126.com
# @File    : tf-stack.py
# @Software: PyCharm
# coding:utf-8
import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8, 9], [0, 1, 7]])
s = tf.constant([[10, 11, 12], [13, 14, 15]])
# c是3*2*3的张量
# tf.stack（）这是一个矩阵拼接的函数
c = tf.stack([a, b, s], axis=0)

# perm=[0,1,2],0代表三维数组的高（即为二维数组的个数），1代表二维数组的行，2代表二维数组的列
# tf.transpose(x, perm=[1,0,2])代表将三位数组的高和行进行转置,
ts = tf.transpose(c, perm=[1, 0, 2])
# tf.unstack（）则是一个矩阵分解的函数
un = tf.unstack(ts)

with tf.Session() as sess:
    print("***c***")
    print(sess.run(c))
    print()

    print("***ts***")
    print(sess.run(ts))
    print()

    print("***un***")
    print(sess.run(un))
    print()


