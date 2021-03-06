from collections import OrderedDict
from mnistlist.common.gradient import numerical_gradient
from common.Layer import *


class TwoLayerNetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init):
        self.parameters = {
            'first_weight': weight_init * np.random.randn(input_size, hidden_size),
            'first_bias': np.zeros(hidden_size),
            'second_weight': weight_init * np.random.randn(hidden_size, output_size),
            'second_bias': np.zeros(output_size)
        }

        # Set Layer
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.parameters['first_weight'], self.parameters['first_bias'])
        self.layers['Relu1'] = ReluFunc()
        self.layers['Affine2'] = Affine(self.parameters['second_weight'], self.parameters['second_bias'])
        self.LastLayer = SoftMaxwithLoss()

    def predict(self, x):
        # first_weight, second_weight = self.parameters['first_weight'], self.parameters['second_weight']
        # first_bias, second_bias = self.parameters['first_bias'], self.parameters['second_bias']
        #
        # layer01 = np.dot(x, first_weight) + first_bias
        # layer01_sigmoid = sigmoid(layer01)
        # layer02 = np.dot(layer01_sigmoid, second_weight) + second_bias
        # output = softmax(layer02)

        for layer in self.layers.values():
            output = layer.forward(x)
        return output

    def get_lossrate(self, x, teacher):
        predict_result = self.predict(x)

        return cross_entropy_error(predict_result, teacher)

    def get_accuracyrate(self, x, teacher):
        predict_result = self.predict(x)
        predict_result_maxargs = np.argmax(predict_result, axis=1)
        teacher_maxargs = np.argmax(teacher, axis=1)

        accuracy_rate = np.sum(predict_result_maxargs == teacher_maxargs) / float(x.shape[0])
        return accuracy_rate

    def get_numerical_gradient(self, x, teacher):
        loss = lambda W: self.get_lossrate(x, teacher)
        grads = {
            'first_weight': numerical_gradient(loss, self.parameters['first_weight']),
            'first_bias': numerical_gradient(loss, self.parameters['first_bias']),
            'second_weight': numerical_gradient(loss, self.parameters['second_weight']),
            'second_bias': numerical_gradient(loss, self.parameters['second_bias'])
        }
        return grads

    def get_fast_gradient(self, x, teacher):
        self.get_lossrate(x, teacher)

        diff = 1
        diff = self.LastLayer.backward(diff)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            diff = layer.backward(diff)

        grads = {
            'first_weight': self.layers['Affine1'].diff_weight,
            'first_bias': self.layers['Affine1'].diff_bias,
            'second_weight': self.layers['Affine2'].diff_weight,
            'second_bias': self.layers['Affine2'].diff_bias
        }

        return grads
