import numpy as np

try:
    import cudamat as cm
    from DBNlogic.util import CUDA as pu
except ImportError:
    from DBNlogic.util import CPU as pu

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
        pu.start()

        net        = self.net
        max_epochs = self.config.max_epochs
        batch_sz   = self.config.batch_size
        learn_rate = self.config.learn_rate
        momentum   = self.config.momentum
        w_decay    = self.config.w_decay

        W_update = np.zeros(net.W.shape)
        a_update = np.zeros(net.a.shape)
        b_update = np.zeros(net.b.shape)

        epoch = 1
        while epoch <= max_epochs:
            errors = np.array([])
            for batch_n in range(int(len(trainset) / batch_sz)):
                start = batch_sz * batch_n
                data = pu.matrix(np.array(trainset[start : start + batch_sz]).T)

                # --- positive phase:
                pos_hid_probs = pu.numpy(pu.sigmoid(pu.add(pu.dot(net.W, data), pu.repeat(net.b, batch_sz, axis = 1))))
                hid_states = pu.numpy(pu.activation(pos_hid_probs))
                pos_corr = pu.numpy(pu.div(pu.dot(pos_hid_probs, data.T), batch_sz)) # vis-hid correlations (+)
                pos_vis_act = pu.div(pu.cumsum(data, axis = 1), batch_sz)
                pos_hid_act = pu.div(pu.cumsum(pos_hid_probs, axis = 1), batch_sz)

                # --- build the training set for the next RBM:
                if epoch == max_epochs:
                    self.next_rbm_data.extend(pos_hid_probs.T)

                # --- negative phase:
                vis_probs = pu.numpy(pu.sigmoid(pu.add(pu.dot(net.W.T, hid_states), pu.repeat(net.a, batch_sz, axis = 1))))
                reconstr = pu.numpy(pu.activation(vis_probs))
                neg_hid_probs = pu.numpy(pu.sigmoid(pu.add(pu.dot(net.W, reconstr), pu.repeat(net.b, batch_sz, axis = 1))))
                neg_corr = pu.numpy(pu.div(pu.dot(neg_hid_probs, reconstr.T), batch_sz)) # vis-hid correlations (-)
                neg_vis_act = pu.numpy(pu.div(pu.cumsum(reconstr, axis = 1), batch_sz))
                neg_hid_act = pu.numpy(pu.div(pu.cumsum(neg_hid_probs, axis = 1), batch_sz))

                # --- updates:
                W_update = pu.add(pu.mul(momentum, W_update), pu.mul(learn_rate, pu.sub(pu.sub(pos_corr, neg_corr), pu.mul(w_decay, net.W))))
                a_update = pu.add(pu.mul(momentum, a_update), pu.mul(learn_rate, pu.sub(pos_vis_act, neg_vis_act)))
                b_update = pu.add(pu.mul(momentum, b_update), pu.mul(learn_rate, pu.sub(pos_hid_act, neg_hid_act)))
                net.W += pu.numpy(W_update)
                net.a += pu.numpy(a_update)
                net.b += pu.numpy(b_update)
                errors = np.append(errors, squared_error(data, reconstr))

            # --- reconstruction error update:
            mean_squared_err = errors.mean()

            epoch += 1
            yield mean_squared_err

        pu.shutdown()
