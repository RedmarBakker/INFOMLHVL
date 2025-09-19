from ConvLayer import ConvLayer


class Model:

    def __init__(self, input_shape):
        self.layers = []
        # self.weights = []
        self.input_shape = input_shape

    def add(self, layer):
        self.layers.append(layer)

    def process(self, input):
        for layer in self.layers:
            input = layer.process(input)

        return input