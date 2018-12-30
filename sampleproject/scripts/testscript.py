# coding: utf-8
import numpy as np
from mnistlist.dataset.mnist import load_mnist
from common.TwoLayerNet import TwoLayerNetwork

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNetwork(input_size=784, hidden_size=50, output_size=10, weight_init=0.01)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.get_numerical_gradient(x_batch, t_batch)
grad_backprop = network.get_fast_gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
