import matplotlib.pyplot as plt



class WeightsPlotter:
    """Plotter for the weights of a neural network layer."""

    def __init__(self, net): # + shape of image, if any! <<<<
        self.net = net

    def weights(self):
        return self.net.W

    def plot(self):
        plt.imshow(self.net.W)
        plt.show()
