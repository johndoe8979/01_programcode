class multiple_layer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        output = x * y
        return output

    def backward(self, differential):
        x_differential = differential * self.y
        y_differential = differential * self.x
        return x_differential, y_differential


class add_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        output = x + y
        return output

    def backward(self, differential):
        differential_x = differential * 1
        differential_y = differential * 1
        return differential_x, differential_y

class relu():

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

    def backward(self, differential):
        differential[self.mask] = 0
        diff_output = differential

        return differential