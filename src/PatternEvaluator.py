# imports

import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError

import time

from SamplingPattern import SamplingPattern
from defaults import base_sz
from utils import ft, ift, ft2, ift2


class PatternEvaluator(object):
    """docstring for PatternEvaluator"""
    def __init__(self, base_sz = BASE_N, sens=[], max_tries=2):
        super(PatternEvaluator, self).__init__()

        # Base size
        self.base_sz = base_sz

        # The ky-kz sampling pattern we want to test.
        self.pattern = None

        self.xnew = zeros((self.base_sz, self.base_sz), dtype='complex')
        self.xm = zeros((self.base_sz, self.base_sz), dtype='complex')
        self.sampling = zeros((self.base_sz, self.base_sz), dtype='float')

        self.max_tries = max_tries


    def set_norm_fac(self, p):
        if hasattr(p, 'norm_fac') and p.norm_fac > 0:
            print 'Using pattern normfac of {}'.format(p.norm_fac)
            self.norm_fac = p.norm_fac
        else:
            self.norm_fac = self.n_coils
            print 'Using normfac of {}'.format(self.norm_fac)


    def eval_pattern(self, p):
        self.pattern = p
        self.sampling = p.sampling.copy().astype('float')

        self.set_norm_fac(p)

        self.solve_high()

        self.solve_low()

        self.pattern.calcd = True

        print p.hi_eigs
        print p.low_eigs

    def solve_high(self):

        t_start = time.time()

        sysA = scipy.sparse.linalg.LinearOperator(
                        (self.mask_sz, self.mask_sz),
                        matvec=self.calc_AtA,
                        dtype='complex')

        solved = False

        for j in range(self.maxTries):
            try:
                a1,v1 = scipy.sparse.linalg.eigsh(
                            sysA,
                            k=self.pattern.n_eigs,
                            which='LM',
                            maxiter=self.pattern.iter_max,
                            tol=self.pattern.hitol,
                            ncv=self.pattern.ncv,
                            return_eigenvectors=True)

                # sometimes it "solves" but with crazy errors ~1e+_300
                if np.any(np.abs(a1) > self.n_coils):
                    continue
                else:
                    solved = True
                    break
            except ArpackError as e:
                if e.info == -8:
                    print('error on try {}'.format(j))


        t_end = time.time()

        print "Elapased: {}s".format(t_end - t_start)
        self.pattern.hi_eigs = a1

        if not solved:
            self.pattern.hi_eigs = -1




    def solve_low(self):

        t_start = time.time()

        sysA = scipy.sparse.linalg.LinearOperator(
                        (self.mask_sz, self.mask_sz),
                        matvec=self.calc_AtA,
                        dtype='complex')

        solved = False

        for j in range(self.maxTries):
            try:
                adyn,vdyn = scipy.sparse.linalg.eigsh(
                                sysA,
                                k=self.pattern.n_eigs,
                                which='SM',
                                maxiter=self.pattern.iter_max,
                                tol=self.pattern.tol,
                                ncv=self.pattern.ncv,
                                return_eigenvectors=True)

                # sometimes it "solves" but with awful numerical problems
                if np.any(np.abs(adyn) > 1e3):
                    continue
                else:
                    solved = True
                    break

            except ArpackError as e:
                print('Arpack error in solve_low {}'.format(e))

        t_end = time.time()

        print "Elapased: {}s".format(t_end - t_start)

        self.pattern.low_eigs = adyn

        if not solved:
            self.pattern.low_eigs = -1

    def calc_AtA(self):
        nSamp = sum(sampling)
        maskSz = sum(mask)
        nCoils, nv, npts = sens.shape
        if x0.dtype <> np.complex128:
            x0 = x0.astype('complex128')
        x_img = x0[0:maskSz]
        x_ksp = x0[maskSz:]

        A_fwd = zeros(nSamp*nCoils, dtype='complex')

        result = zeros(maskSz, dtype='complex')

        # Compute A
        sys_fwd(x_img, A_fwd, sens, sampling>0, mask)

        #compute A^H
        A_back = sys_bck(A_fwd, sens, sampling>0, mask)
        result[:] = A_back[:]

        return result


def sys_fwd(im_mask, data, coils, pattern, mask):
    nCoils, nv, npts  = coils.shape
    #print coils.shape
    #print data.shape
    image = zeros((nv, npts), dtype='complex128')
    image[mask] = im_mask
    data.shape= (nCoils,-1)
    nD = image.ndim
    accum = 0.0
    tmpGrad = []

    #for each coil, do S^H . I
    for c in range(nCoils):
        coilPtr = coils[c,...]
        dataPtr = data[c,...]

        # todo: zeropad
        scratch = (coilPtr) * image
        scratch = ift2(scratch)

        s_reduced = scratch[pattern]
        data[c,...] =  s_reduced[:]
        #print sum(abs(scratch.ravel()))


    data.shape=(-1)




#back
def sys_bck(data, coils, pattern, mask):
    nCoils, nv, npts  = coils.shape
    #print coils.shape
    #print data.shape
    data.shape= (nCoils,-1)
    nD = coils[0].ndim
    accum = 0.0
    tmpGrad = []
    maskSz = sum(mask)
    gradient = zeros(maskSz)
    scratch = zeros_like(coils[0])

    #for each coil, do S^H . I
    for c in range(nCoils):
        coilPtr = coils[c,...]
        dataPtr = data[c,...]

        scratch = scratch*0

        scratch[pattern] = dataPtr[:]

        scratch = ft2(scratch)
        # todo: crop
        scratch = conj(coilPtr) * scratch


        gradient = gradient + scratch[mask]
        #print sum(abs(scratch.ravel()))


    data.shape = (-1)
    gout = (gradient)
    gout.shape = (-1)
    return gout


