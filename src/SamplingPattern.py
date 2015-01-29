
import numpy as np

from defaults import BASE_N


# class for sampling pattern. we're going to save some properties so we can
# regrenerate pattern also compute eigs and save

class SamplingPattern():

    def __init__(self, nx=BASE_N, ny=BASE_N, s_name='pattern_default'):

        # dimensions
        self.nx = nx
        self.ny = ny
        self.s_name = s_name

        # xvec and yvec for 'regular' outer product sampling schemes
        self.xvec = np.ones((1, nx))
        self.yvec = np.ones((ny, 1))

        # flags for calculation. Sampling pattern remembers the params
        # used to determine its eigs
        self.calcd = False
        self.failed = False

        self.n_eigs = 1
        self.tol = 1e-7             # tol for computing low eigs
        self.hitol = 1e-3           # to for hi eigs
        self.ncv = 100              # ncv for arpack
        self.iter_max = 8000
        self.low_eigs = []
        self.hi_eigs = []
        self.sample_pct = 0.0        # 1/reduction factor
        self.reg_lambda = 0.0       # regularization param (future use)

        # for display
        self.min_low = 0.0
        self.max_hi = 0.0
        self.ratio = -1.0

    def calc_pattern(self):
        """ create a sampling pattern with given size/sampling.
         override me!
         Default beahviour: take outer product of xvec and yvec
         Compute reduction factor for later display.
         """

        pattern = np.dot(self.yvec, self.xvec)

        # Float needed for ctypes conversion
        pattern = pattern.astype('float')

        self.sample_pct = sum(pattern) / (np.prod(pattern.shape) * 1.0)
        return pattern

    def calc_min_max(self):
        """ pull out min/max eig + ratio for display """
        if self.calcd:
            self.max_hi = np.max(abs(self.hi_eigs))
            self.min_low = np.min(abs(self.low_eigs))
            self.ratio = self.max_hi / self.min_low

    def str_eigs(self):
        tmp = "{:23} l:{:6}  R{:2} r: {:8.3}  min: {:10.3}  max: {:10.3}".format(self.s_name, self.reg_lambda, self.low_eigs[0], self.hi_eigs[0])
        return tmp

    def print_max_min(self):
        self.calc_min_max()
        tmp = "{}/{}\n\t \t{} \t: \t{}  \n".format(self.s_name, self.reg_lambda, self.max_hi, self.min_low)
        return tmp

    def __repr__(self):
        return self.s_name

