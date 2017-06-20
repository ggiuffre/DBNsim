import numpy as np

from DBNlogic.util import Configuration

USE_GPU = False
try:
    from DBNlogic.gpu import matrix, asnumpy, transpose, dot, div, mul, add, sub, cumsum, repeat, sigmoid, activation, squared_error
    USE_GPU = True
    import cudamat as cm
except ImportError:
    from DBNlogic.util import matrix, asnumpy, transpose, dot, div, mul, add, sub, cumsum, repeat, sigmoid, activation, squared_error



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, config = Configuration()):
        """Construct a strategy object for training a RBM
        with the Contrastive Divergence learning algorithm."""
        self.net = net          # the learner
        self.config = config    # hyper-parameters for training
        self.next_rbm_data = [] # the training set built from the positive probs

        self.train_manually = False # hand training:
        # set to True when you want to train the network indefinitely;
        # then set to False when you're done with the training.

    def run(self, trainset):
        """Learn from a particular dataset."""
        net        = self.net
        max_epochs = self.config.max_epochs
        batch_sz   = self.config.batch_size
        learn_rate = self.config.learn_rate
        momentum   = self.config.momentum
        w_decay    = self.config.w_decay

        W_update = matrix(np.zeros(net.W.shape))
        a_update = matrix(np.zeros(net.a.shape))
        b_update = matrix(np.zeros(net.b.shape))

        epoch = 1
        while (epoch <= max_epochs) or self.train_manually:
            errors = np.array([])
            for batch_n in range(int(len(trainset) / batch_sz)):
                start = batch_sz * batch_n
                data = matrix(np.array(trainset[start : start + batch_sz]).T)

                # --- positive phase:
                if USE_GPU:
                    pos_hid_probs = cm.sigmoid(cm.dot(cm.CUDAMatrix(net.W), data).add(cm.dot(cm.CUDAMatrix(net.b), cm.CUDAMatrix(np.ones((1, batch_sz))))))
                    hid_states = pos_hid_probs.subtract(cm.CUDAMatrix(np.random.uniform(size = pos_hid_probs.shape))).sign().add(cm.CUDAMatrix(np.ones((pos_hid_probs.shape)))).divide(cm.CUDAMatrix(2 * np.ones((pos_hid_probs.shape))))
                    pos_corr = cm.dot(pos_hid_probs, data.transpose()).divide(batch_sz) # vis-hid correlations (+)
                    pos_vis_act = cm.sum(data, axis = 1).divide(batch_sz)
                    pos_hid_act = cm.sum(pos_hid_probs, axis = 1).divide(batch_sz)
                else:
                    pos_hid_probs = sigmoid(add(dot(net.W, data), repeat(net.b, batch_sz, axis = 1)))
                    hid_states = activation(pos_hid_probs)
                    pos_corr = div(dot(pos_hid_probs, data.T), batch_sz) # vis-hid correlations (+)
                    pos_vis_act = div(cumsum(data, axis = 1), batch_sz)
                    pos_hid_act = div(cumsum(pos_hid_probs, axis = 1), batch_sz)

                # --- build the training set for the next RBM:
                if epoch == max_epochs:
                    self.next_rbm_data.extend(asnumpy(pos_hid_probs).T)

                # --- negative phase:
                if USE_GPU:
                    vis_probs = cm.sigmoid(cm.dot(cm.CUDAMatrix(net.W).transpose(), hid_states).add(cm.dot(cm.CUDAMatrix(net.a), cm.CUDAMatrix(np.ones((1, batch_sz))))))
                    reconstr = vis_probs.subtract(cm.CUDAMatrix(np.random.uniform(size = vis_probs.shape))).sign().add(cm.CUDAMatrix(np.ones((vis_probs.shape)))).divide(cm.CUDAMatrix(2 * np.ones((vis_probs.shape))))
                    neg_hid_probs = cm.sigmoid(cm.dot(cm.CUDAMatrix(net.W), reconstr).add(cm.dot(cm.CUDAMatrix(net.b), cm.CUDAMatrix(np.ones((1, batch_sz))))))
                    neg_corr = cm.dot(neg_hid_probs, reconstr.transpose()).divide(batch_sz) # vis-hid correlations (-)
                    neg_vis_act = cm.sum(reconstr, axis = 1).divide(batch_sz)
                    neg_hid_act = cm.sum(neg_hid_probs, axis = 1).divide(batch_sz)
                else:
                    vis_probs = sigmoid(add(dot(net.W.T, hid_states), repeat(net.a, batch_sz, axis = 1)))
                    reconstr = activation(vis_probs)
                    neg_hid_probs = sigmoid(add(dot(net.W, reconstr), repeat(net.b, batch_sz, axis = 1)))
                    neg_corr = div(dot(neg_hid_probs, reconstr.T), batch_sz) # vis-hid correlations (-)
                    neg_vis_act = div(cumsum(reconstr, axis = 1), batch_sz)
                    neg_hid_act = div(cumsum(neg_hid_probs, axis = 1), batch_sz)

                # --- updates:
                if USE_GPU:
                    W_update = W_update.mult(momentum).add(pos_corr.subtract(neg_corr).subtract(cm.CUDAMatrix(net.W).mult(w_decay)).mult(learn_rate))
                    a_update = a_update.mult(momentum).add(pos_vis_act.subtract(neg_vis_act).mult(learn_rate))
                    b_update = b_update.mult(momentum).add(pos_hid_act.subtract(neg_hid_act).mult(learn_rate))
                else:
                    W_update = add(mul(momentum, W_update), mul(learn_rate, sub(sub(pos_corr, neg_corr), mul(w_decay, net.W))))
                    a_update = add(mul(momentum, a_update), mul(learn_rate, sub(pos_vis_act, neg_vis_act)))
                    b_update = add(mul(momentum, b_update), mul(learn_rate, sub(pos_hid_act, neg_hid_act)))
                net.W += asnumpy(W_update)
                net.a += asnumpy(a_update)
                net.b += asnumpy(b_update)
                errors = np.append(errors, asnumpy(squared_error(data, reconstr)))

            # --- reconstruction error update:
            mean_squared_err = errors.mean()

            epoch += 1
            yield mean_squared_err
