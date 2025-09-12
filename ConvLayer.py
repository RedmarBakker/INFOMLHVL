

class ConvLayer:

    def __init__(self, filters=None, kernel_size=(1, 1), stride=(1, 1)):
        if filters is None:
            filters = []

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride

    def add_filter(self, parameters=None):
        self.filters.append(parameters)

    def process(self, input):


        return input