import numpy as np
from nets import DBN, RBM
from plot import WeightsPlotter
from sets import MNIST



def main():
    L1 = RBM(784, 24)
    net = DBN([L1])
    
    trainset = MNIST()
    examples = np.array([ex['image'] for ex in trainset.data]).reshape(60000, 784)

    net.learn(examples[:1], max_epochs = 10)
    # for test in validation:
    #     print(test, '-->', net.evaluate(test))
    
    import matplotlib.pyplot as plt
    plt.imshow(net.layers[0].W.reshape((len(net.layers[0].h), 28, 28))[0])
    plt.show()



if __name__ == '__main__':
    main()
