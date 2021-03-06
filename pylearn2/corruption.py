"""
Corruptor classes: classes that encapsulate the noise process for the DAE
training criterion.
"""
# Third-party imports
import numpy
import theano
from theano import tensor
T = tensor
from theano.printing import Print
# Shortcuts
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Corruptor(object):
    def __init__(self, corruption_level, rng=2001):
        """
        Allocate a corruptor object.

        Parameters
        ----------
        corruption_level : float
            Some measure of the amount of corruption to do. What this
            means will be implementation specific.
        rng : RandomState object or seed
            NumPy random number generator object (or seed for creating one)
            used to initialize a RandomStreams.
        """
        # The default rng should be build in a deterministic way
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
        self.corruption_level = corruption_level

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a noise process.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs.

        Notes
        -----
        In the base class, this is just a stub.
        """
        raise NotImplementedError()

    def corruption_free_energy(self, corrupted_X, X):
        raise NotImplementedError()


class DummyCorruptor(Corruptor):
    def __call__(self, inputs):
        return inputs


class BinomialCorruptor(Corruptor):
    """
    A binomial corruptor sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """
    def _corrupt(self, x):
        return self.s_rng.binomial(
            size=x.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * x

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a binomial (masking) noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted inputs,
            where individual inputs have been masked with independent
            probability equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        else:
            return [self._corrupt(inp) for inp in inputs]


class GaussianCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero
    mean isotropic Gaussian noise.
    """

    def __init__(self, stdev, rng=2001):
        super(GaussianCorruptor, self).__init__(corruption_level=stdev, rng=rng)

    def _corrupt(self, x):
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        rval = noise + x

        return rval

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with Gaussian noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs,
            where individual inputs have been corrupted by zero mean Gaussian
            noise with standard deviation equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        return [self._corrupt(inp) for inp in inputs]

    def corruption_free_energy(self, corrupted_X, X):
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval


class MultivariateGaussianCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero
    mean isotropic Gaussian noise.
    """

    def _exp_cov_cholesky(self, p, sigma=1.0):
        np = numpy
        sigma2 = sigma**2
        C = np.zeros((p,p))
        x = np.exp(-1./sigma2)
        c = np.sqrt(1.0-x**2)
        for i in xrange(p):
            C[i,0] = x**i/c
            for j in xrange(1, i+1):
                C[i,j] = x**(i-j)
        C *= c
        return C

    def _smooth_diag(self, p, corruption_level, sigma):
        self.mu = numpy.zeros(p)
        #self.cov = numpy.ones((p,p))

        #for i in xrange(p):
        #    for j in xrange(p):
        #        self.cov[i,j] = numpy.exp(-(j-i)**2/(sigma**2))
        #self.cov *= corruption_level
        #self.L = theano.shared(numpy.linalg.cholesky(self.cov))
        self.L = theano.shared(corruption_level * self._exp_cov_cholesky(p, sigma).T.copy())


    def __init__(self, p, corruption_level=0.5, sigma=10., rng=2001):
        super(MultivariateGaussianCorruptor, self).__init__(corruption_level=0.0, rng=rng)
        self._smooth_diag(p, corruption_level, sigma)
        #self.mu = mu
        #self.cov = cov

    def _corrupt(self, x):
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0,
            std=1.0,
            dtype=theano.config.floatX
        )
        #noise.eval()#{x:theano.tensor.as_tensor_variable(numpy.ones((50,1)))})

        print x.dtype
        rval = noise + theano.dot(x, self.L)

        return rval

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with Gaussian noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs,
            where individual inputs have been corrupted by zero mean Gaussian
            noise with standard deviation equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        return [self._corrupt(inp) for inp in inputs]


class GaussianScaleCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero
    mean isotropic Gaussian noise.
    """

    def __init__(self, stdev, rng=2001):
        super(GaussianScaleCorruptor, self).__init__(corruption_level=stdev, rng=rng)

    def _corrupt(self, x):
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        scale = numpy.random.rand()

        rval = scale * noise + x

        return rval

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with Gaussian noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs,
            where individual inputs have been corrupted by zero mean Gaussian
            noise with standard deviation equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        return [self._corrupt(inp) for inp in inputs]


##################################################
def get(str):
    """ Evaluate str into a corruptor object, if it exists """
    obj = globals()[str]
    if issubclass(obj, Corruptor):
        return obj
    else:
        raise NameError(str)
