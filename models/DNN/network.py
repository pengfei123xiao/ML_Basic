#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/09/2019 5:32 PM
# @Author  : Pengfei Xiao
# @FileName: network.py
# @Software: PyCharm
from models.DNN.full_connect import FullConnectedLayer
from models.DNN.activator import ReluActivator, SigmoidActivator, TanhActivator


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """构造函数"""
        self.history = {}
        # self.history['output_delta'] = []
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1],
                    # TanhActivator()
                    # ReluActivator()
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
            self.history[f'fc{self.layers.index(layer)+1}_W'] = layer.W
            self.history[f'fc{self.layers.index(layer)+1}_b'] = layer.b
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)
        return self.history

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        # 式7
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        # self.history['output_delta'].append(delta)
        self.history['output_delta'] = delta
        for layer in self.layers[::-1]:  # 反向遍历
            layer.backward(delta)
            delta = layer.delta
            self.history[f'fc{self.layers.index(layer)+1}_delta'] = delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
            # self_W, self_b = layer.update(rate)
            # self.history[f'fc{self.layers.index(layer)+1}_W'] = self_W
            # self.history[f'fc{self.layers.index(layer)+1}_b'] = self_b
