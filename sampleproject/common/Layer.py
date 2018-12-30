import numpy as np
from mnistlist.common.functions import *
from mnistlist.common.util import im2col, col2im


class ReluFunc:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        output = x.copy()
        output[self.mask] = 0

        return output

    def backward(self, diff_out):
        diff_out[self.mask] = 0
        diff_back = diff_out

        return diff_back


class SigmoidFunc:

    def __init__(self):
        self.out = None

    def forward(self, x):
        output = sigmoid(x)
        self.out = output

        return output

    def backward(self, diff_out):
        diff_back = diff_out * self.out * (1 - self.out)

        return diff_back


class Affine:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.x = None
        self.original_x_shape = None

        self.diff_weight = None
        self.diff_bias = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        output = np.dot(self.x, self.weight) + self.bias

        return output

    def backward(self, diff_out):
        diff_x = np.dot(diff_out, self.weight.T)
        self.weight = np.dot(self.x.T, diff_out)
        self.bias = np.sum(diff_out, axis=0)
        diff_x = diff_x.reshape(*self.original_x_shape)

        return diff_x


class SoftMaxwithLoss:

    def __init__(self):
        self.loss = None
        self.softmax_output = None
        self.teacher = None

    def forward(self, x, teacher):
        self.teacher = teacher
        self.softmax_output = softmax(x)
        self.loss = cross_entropy_error(self.softmax_output, self.teacher)

        return self.loss

    def backward(self, diff_out):
        batch_size = self.teacher.shape[0]

        if self.teacher.size == self.softmax_output:
            diff_x = (self.softmax_output - self.teacher) / batch_size
        else:
            diff_x = self.softmax_output.copy()
            diff_x[np.array(batch_size), self.teacher] -= 1
            diff_x = diff_x / batch_size

        return diff_x
