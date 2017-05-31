import numpy as np

from DBNlogic.util import Configuration, sigmoid, activation, squared_error



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, config = Configuration()):
        """Construct a strategy object for training a RBM
        with the Contrastive Divergence learning algorithm."""
        self.net = net          # the learner
        self.config = config    # hyper-parameters for training
        self.next_rbm_data = [] # the training set built from the positive probs

    def run(self, trainset):
        """Learn from a particular dataset."""
        net        = self.net
        max_epochs = self.config.max_epochs
        threshold  = self.config.threshold
        batch_sz   = self.config.batch_size
        learn_rate = self.config.learn_rate
        momentum   = self.config.momentum
        w_decay    = self.config.w_decay

        W_update = np.zeros(net.W.shape)
        a_update = np.zeros(net.a.shape)
        b_update = np.zeros(net.b.shape)

        epoch = 1
        mean_squared_err = threshold + 1 # (for entering the while loop)
        while (mean_squared_err > threshold) and (epoch <= max_epochs):
            errors = np.array([])
            for batch_n in range(int(len(trainset) / batch_sz)):
                start = batch_sz * batch_n
                data = np.array(trainset[start : start + batch_sz]).T

                # --- positive phase:
                pos_hid_probs = sigmoid(np.dot(net.W, data) + net.b.repeat(batch_sz, axis = 1))
                hid_states = activation(pos_hid_probs)
                pos_corr = np.dot(pos_hid_probs, data.T) / batch_sz # vis-hid correlations (+)
                pos_vis_act = data.sum(axis = 1, keepdims = True) / batch_sz
                pos_hid_act = pos_hid_probs.sum(axis = 1, keepdims = True) / batch_sz

                # --- build the training set for the next RBM:
                if epoch == max_epochs:
                    self.next_rbm_data.extend(pos_hid_probs.T)
                    print('next RBM data length:', len(self.next_rbm_data))

                # --- negative phase:
                vis_probs = sigmoid(np.dot(net.W.T, hid_states) + net.a.repeat(batch_sz, axis = 1))
                reconstr = activation(vis_probs)
                neg_hid_probs = sigmoid(np.dot(net.W, reconstr) + net.b.repeat(batch_sz, axis = 1))
                neg_corr = np.dot(neg_hid_probs, reconstr.T) / batch_sz # vis-hid correlations (-)
                neg_vis_act = reconstr.sum(axis = 1, keepdims = True) / batch_sz
                neg_hid_act = neg_hid_probs.sum(axis = 1, keepdims = True) / batch_sz

                # --- updates:
                W_update = momentum * W_update + learn_rate * ((pos_corr - neg_corr) - w_decay * net.W)
                a_update = momentum * a_update + learn_rate * (pos_vis_act - neg_vis_act)
                b_update = momentum * b_update + learn_rate * (pos_hid_act - neg_hid_act)
                net.W += W_update
                net.a += a_update
                net.b += b_update
                errors = np.append(errors, squared_error(data, reconstr))

            # --- error update:
            mean_squared_err = errors.mean()

            epoch += 1
            yield mean_squared_err
