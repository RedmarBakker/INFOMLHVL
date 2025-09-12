from ConvLayer import Layer


class Model:

    layers = []
    weights = []

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def add(self, layer):
        self.layers.append(layer)

    def process(self, input):
        for layer in self.layers:
            input = layer.process(input)

        return input