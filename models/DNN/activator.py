#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/09/2019 5:31 PM
# @Author  : Pengfei Xiao
# @FileName: activator.py
# @Software: PyCharm

import numpy as np


# 激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)  # gradient


class ReluActivator(object):
    def forward(self, weighted_input):
        return np.maximum(0, weighted_input)

    def backward(self, output):
        return 1.0 * (output > 0) # gradient: if output > 0: 1 else: 0

class TanhActivator(object):
    def forward(self, weighted_input):
        return np.tanh(weighted_input)

    def backward(self, output):
        return (1-np.square(output))