from AllLayers import ConvLayer
from Layer import Layer

class Model:

    def __init__(self):
        self.layers = []
        self.weights = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def process(self, input):
        for layer in self.layers:
            input = layer.process(input)

        return input