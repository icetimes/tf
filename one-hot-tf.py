# -*- coding: utf-8 -*-
# @Time    : 2019/4/30 9:58
# @Author  : huyulan
# @Email   : huyulan@boe.com.cn
# @File    : one-hot-tf.py
# @Software: PyCharm

import tensorflow as tf

# 转换为独热编码
classes = 3
labels = tf.constant([0,1,2, 0, 2]) # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, classes)

sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(output)
    print(output)