import numpy


class Perceptron:
    def __init__(self, size, weights=None):
        self.size = size
        self.weights = numpy.array(weights)
        # init weights = array zeros when not provide
        if weights is None or len(weights) != size:
            self.weights = numpy.zeros(size)

    def predict(self, features):
        score = numpy.dot(features, self.weights)
        return score

    def adjust(self, adjust_weights):
        self.weights += adjust_weights
