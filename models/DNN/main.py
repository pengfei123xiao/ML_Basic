#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/09/2019 5:35 PM
# @Author  : Pengfei Xiao
# @FileName: main.py
# @Software: PyCharm
from datetime import datetime
from models.DNN.mnist_loader import MnistLoader
from models.DNN.network import Network


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        # label = get_result(test_labels[i])
        label = test_labels[i]
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def now():
    return datetime.now().strftime('%c')


if __name__ == '__main__':
    mnist_loader = MnistLoader()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    train_data_set, train_labels = mnist_loader.get_training_data_set(300, training_data)
    test_data_set, test_labels = mnist_loader.get_test_data_set(100, test_data)

    last_error_ratio = 1.0
    epoch = 0
    learn_rate = 0.01
    network = Network([784, 300, 10])
    while epoch < 3:  # True:
        epoch += 1
        history = network.train(train_labels, train_data_set, learn_rate, 1)
        print('%s epoch %d finished' % (now(), epoch))
        if epoch % 3 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            # when error_ratio stop to decrease, break loop
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio
