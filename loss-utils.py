# -*- coding: utf-8 -*-
# @Time    : 2019/5/5 14:08
# @Author  : huyulan
# @Email   : huyulan@boe.com.cn
# @File    : loss.py
# @Software: PyCharm

import tensorflow as tf


def cross_entropy(y_pred, y_true):
    vv = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    return vv


if __name__ == '__main__':

    with tf.Session() as sess:
        y_pred = [0.1, 0.5]
        y_true = [1, 0]
        vv = cross_entropy(y_pred, y_true)

        print(sess.run(vv))


