import matplotlib.pyplot as plt



class WeightsPlotter:
    """Plotter for the weights of a neural network layer."""

    def __init__(self, net): # + shape of image, if any! <<<<
        self.net = net

    def weights(self):
        return self.net.W

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        plt.imshow(self.net.W)#, interpolation = 'nearest', cmap = plt.cm.ocean)
        plt.colorbar()
        plt.show()
