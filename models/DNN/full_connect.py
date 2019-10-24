#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/09/2019 5:18 PM
# @Author  : Pengfei Xiao
# @FileName: full_connect.py
# @Software: PyCharm
import numpy as np
import math

# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
                                   (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)  # 对于w，delta*x是梯度
        self.b_grad = delta_array  # 对于b，delta是梯度

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad
        return self.W, self.b

    def adam_update(self, learning_rate):
        def _adam(g_ts):
            # g_ts: [[n1...nn],[n1...nn],...,[n1...nn]] n_label * n_samples
            for rows in g_ts:
                row, col = 0, 0
                # initialize the values of the parameters
                beta_1 = 0.9
                beta_2 = 0.999
                epsilon = 1e-8
                # initialize the vector
                theta_0, m_t, v_t, t = 0, 0, 0, 0
                for g_t in rows:  # traverse each column
                    while True:
                        t += 1
                        m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
                        v_t = beta_2 * v_t + (1 - beta_2) * (
                                    g_t * g_t)  # updates the moving averages of the squared gradient
                        m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
                        v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
                        theta_0_prev = theta_0
                        theta_0 = theta_0 - (learning_rate * m_cap) / (
                                    math.sqrt(v_cap) + epsilon)  # updates the parameters
                        if abs(theta_0 - theta_0_prev) < 1e-6:  # checks if it is converged or not
                            g_ts[row, col] = np.float(theta_0)
                            col += 1
                            break
                row += 1
            return g_ts

        self.W = _adam(self.W_grad)
        self.b = _adam(self.b_grad)
