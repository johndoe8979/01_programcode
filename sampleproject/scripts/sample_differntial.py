import numpy as np
from common.MulLayer import *
from collections import OrderedDict

# Set variable
apple = 100
apple_num = 2
tax = 1.1


mulapple_layer = multiple_layer()
multax_layer = multiple_layer()

# forward calc
apple_price = mulapple_layer.forward(apple, apple_num)
price = multax_layer.forward(apple_price, tax)

# backward calc
differential_output = 1
differential_price, differential_tax = multax_layer.backward(differential_output)
differential_apple, differential_applenum = mulapple_layer.backward(differential_price)

print(differential_apple, differential_applenum, differential_tax)


relu_layer = relu()
x = np.array([[-1, 12, 5], [-12, -23, 6]])
relu_layer.forward(x)