import numpy as np
import gnumpy as gpu

from .util import Configuration



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, config = Configuration()):
        """Construct a strategy object for training a RBM
        with the Contrastive Divergence learning algorithm."""

        # The learner.
        self.net = net

        # Hyper-parameters for training.
        self.config = config

        # This is the training set which will be built from the
        # positive probabilities; it will be used for training
        # the next RBM in the DBN hierarchy.
        self.next_rbm_data = None

        # Set to True for manually stopping the training.
        self.handbrake = False

    def run(self, trainset):
        """Learn from a particular dataset."""

        net        = self.net
        max_epochs = self.config.max_epochs
        batch_sz   = self.config.batch_size
        learn_rate = self.config.learn_rate
        momentum   = self.config.momentum
        w_decay    = self.config.w_decay

        gpu.board_id_to_use = 0
        gpu.max_memory_usage = 480 * 1000 * 1000 # not sure...

        net.W = gpu.garray(net.W) # move net.W to the GPU RAM
        net.a = gpu.garray(net.a) # move net.a to the GPU RAM
        net.b = gpu.garray(net.b) # move net.b to the GPU RAM

        W_update = gpu.zeros(net.W.shape)
        a_update = gpu.zeros(net.a.shape)
        b_update = gpu.zeros(net.b.shape)

        if type(trainset) != gpu.garray:
            trainset = gpu.garray(trainset)
        self.next_rbm_data = gpu.zeros((trainset.shape[0], net.W.shape[0]))

        epoch = 1
        while (epoch <= max_epochs) and (not self.handbrake):
            errors = np.array([])
            for batch_n in range(int(len(trainset) / batch_sz)):
                start = batch_n * batch_sz
                data = trainset[start : start + batch_sz].T

                # --> positive phase:
                pos_hid_probs = (gpu.dot(net.W, data) + net.b.tile(batch_sz)).logistic()
                hid_states = pos_hid_probs > pos_hid_probs.rand()
                pos_corr = gpu.dot(pos_hid_probs, data.T) # vis-hid correlations (+)
                pos_vis_act = data.sum(axis = 1)
                pos_hid_act = pos_hid_probs.sum(axis = 1)

                # --> build the training set for the next RBM:
                if epoch == max_epochs:
                    self.next_rbm_data[start : start + batch_sz] = pos_hid_probs.T

                # --> negative phase:
                vis_probs = (gpu.dot(net.W.T, hid_states) + net.a.tile(batch_sz)).logistic()
                reconstr = vis_probs > vis_probs.rand()
                neg_hid_probs = (gpu.dot(net.W, reconstr) + net.b.tile(batch_sz)).logistic()
                neg_corr = gpu.dot(neg_hid_probs, reconstr.T) # vis-hid correlations (-)
                neg_vis_act = reconstr.sum(axis = 1)
                neg_hid_act = neg_hid_probs.sum(axis = 1)

                # --> updates:
                W_update = momentum * W_update + learn_rate * ((pos_corr - neg_corr) / batch_sz - (w_decay * net.W))
                a_update = (momentum * a_update) + (pos_vis_act - neg_vis_act).reshape(-1, 1) * (learn_rate / batch_sz)
                b_update = (momentum * b_update) + (pos_hid_act - neg_hid_act).reshape(-1, 1) * (learn_rate / batch_sz)
                net.W += W_update
                net.a += a_update
                net.b += b_update
                mean_err = gpu.sqrt(((data - reconstr) ** 2).mean())
                errors = np.append(errors, mean_err)

            # --> reconstruction error update:
            mean_squared_err = errors.mean()

            epoch += 1
            yield mean_squared_err

        net.W = net.W.as_numpy_array() # move net.W back to the host's RAM
        net.a = net.a.as_numpy_array() # move net.a back to the host's RAM
        net.b = net.b.as_numpy_array() # move net.b back to the host's RAM
