import numpy as np
import theano
import theano.tensor as T

from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm


class LBFGS(TrainingAlgorithm):
    def __init__(self, cost,
        termination_criterion=None):
        self.cost = cost
        self.termination_criterion = termination_criterion

    def setup(self, model, dataset):
        self.model = model
        #XXX: why needed?
        self.monitor = Monitor.get_monitor(model)
        self.grads, self.grad_updates = self.cost.get_gradients(model, theano.shared(dataset.X))
        self.grad = lambda w: self.grads.values()

        # Convert input

    def train(self, dataset):
        # Call LBFGS
        self.XS = theano.shared(dataset.X)
        self.costfn = self.cost(self.model, self.XS)
        vec = np.ones(29144)

        #from scipy.optimize import fmin_bfgs
        from scipy.optimize import fmin_l_bfgs_b
        vecstar = fmin_l_bfgs_b(f,x0=self.model_to_vector(), fprime=fprime, args=(self,), factr=1e5)

        opt = vecstar[0]
        self.update_model(opt)
        from pylearn2.gui import get_weights_report
        pv = get_weights_report.get_weights_report(model=self.model)
        pv.save('output.png')


        fprime(vec, self)
        1/0
        pass

    def continue_learning(self, model):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion(self.model)

    def update_model(self, vec):
        idx = 0
        vbsize = self.model.visbias.get_value(borrow=True).size
        self.model.visbias.set_value(vec[:vbsize])
        idx += vbsize

        hbsize = self.model.hidbias.get_value(borrow=True).size
        self.model.hidbias.set_value(vec[idx:idx+hbsize])

        wshape = self.model.weights.get_value(borrow=True).shape
        self.model.weights.set_value(np.reshape(vec[idx+hbsize:], wshape))

    def model_to_vector(self):
        vb = self.model.visbias.get_value(borrow=True).flatten()
        hb = self.model.hidbias.get_value(borrow=True).flatten()
        weights = self.model.weights.get_value(borrow=True).flatten()
        return np.concatenate((vb, hb, weights))


def params_to_vector(params):
    wt = ()
    for param in params.values():
        wt = wt + (param.eval().flatten(),)#.get_value(borrow=True))
    return np.concatenate(wt)

def vector_to_params(vec, model):
    pass

def f(w, lbfgs):
    ''' takes weight, computes cost'''
    print '.'
    lbfgs.update_model(w)
    cost = lbfgs.costfn.eval()
    print 'cost =', cost
    return cost

def fprime(w, lbfgs):
    print '\''
    lbfgs.update_model(w)
    grad = params_to_vector(lbfgs.grads)
    print 'grad = ', grad
    return grad


