# imports

import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import ArpackError

import time

from SamplingPattern import SamplingPattern
from defaults import base_sz
from utils import ft, ift, ft2, ift2, sumsq


class PatternEvaluator(object):
    """
    PatternEvaluator
    
    Co-ordinates computation of max and min singular values associated with
    a given SamplingPattern of k-space sample loci.
    
    """
    
    def __init__(self, base_sz = BASE_N, sens=[], max_tries=2):
        super(PatternEvaluator, self).__init__()

        # Base size
        self.base_sz = base_sz

        # SamplingPattern instance we want to test.
        self.pattern = None

        # space for the vectors we need
        self.xnew = np.zeros((self.base_sz, self.base_sz), dtype='complex')
        self.xm = np.zeros((self.base_sz, self.base_sz), dtype='complex')
        # actual array of sampling loci. 
        self.sampling = np.zeros((self.base_sz, self.base_sz), dtype='float')

        # max repeats in case of arpack numerical problems
        self.max_tries = max_tries
        
        if sens:
            self.sens = sens

    def load_sens(self, fname, mask_eps=1e-6):
        """
        load coil sensitivity and masking info from file. 
        Looking for numpy npz file with variable 'sens' 
        Mask from sqrt-sum-of-squares of coil maps.
        """
        fdat = np.load(fname)
        
        #except error
        
        self.sens = fdat['sens'].copy()
        
        ss = sumsq(self.sens)
        
        self.mask = ss > mask_eps
        self.mask_sz = np.sum(self.mask.ravel())
        

    def set_norm_fac(self, p):
        """
        Adjust normalization factor. Used for testing overall
        scaling behaviour of the system.
        Use n_coils.
        """
        if hasattr(p, 'norm_fac') and p.norm_fac > 0:
            print 'Using pattern normfac of {}'.format(p.norm_fac)
            self.norm_fac = p.norm_fac
        else:
            self.norm_fac = self.n_coils
            print 'Using normfac of {}'.format(self.norm_fac)


    def eval_pattern(self, p):
        """
        Main driver routine.
        """
        self.pattern = p
        self.sampling = p.sampling.copy().astype('float')

        self.set_norm_fac(p)

        self.solve_high()

        self.solve_low()

        self.pattern.calcd = True

        print p.hi_eigs
        print p.low_eigs


    def solve_high(self):
        """
        co-ordinate calling ARPACK with our linear operator and get largest eigs
        """

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
                # this seems to be a function of a bad input vector, and typically 
                # is resolved by just running again. if we re-implement arpack
                # we could probably find out why, but until then, we just check for
                # strange values and re-compute.
                if np.any(np.abs(adyn) > 1e3):  # much bigger than nCoils ever will be
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

    def calc_AtA(self, x0):
        """ 
        calculate system matrix (normal equations) 
        """
        nSamp = np.sum(self.sampling)
        maskSz = np.sum(self.mask)
        nCoils, nv, npts = self.sens.shape
        if x0.dtype <> np.complex128:
            x0 = x0.astype('complex128')
        x_img = x0[self.mask]

        result = np.zeros(maskSz, dtype='complex')

        # Compute A
        A_back = sys_sense(x_img, self.sens, self.sampling>0, self.mask)

        result[:] = A_back[:] #copy / flatten

        return result


## --
# Rountines for the system matrix are below.
# To speed things up, we implement these python prototypes in C
#
# Note: fun testing w/ auto-jitting does little here.
#
# Interleaving of the FFT's and dot products are the main slowdown.
# Interestingly, python's default fftpack doesn't do a stellar job
# if we pass in a 3D array and ask for the 2D FT... We can look to move
# to a fftw wrapper in future.
#
# Instead, we overload PatternEvaluator.calc_AtA() to call some
# C functions via the CFFI that do fast dots and call FFTW.
# Its a bit messier for distribution since it requries compilation.
def sys_sense(im_mask, coils, pattern, mask):
    """
    linear system to go from image --> kspace
    """
    nCoils, nv, npts  = coils.shape
    #print coils.shape
    #print data.shape
    image = np.zeros((nv, npts), dtype='complex128')
    image[mask] = im_mask
    nD = image.ndim
    accum = 0.0
    tmpGrad = []
    
    zeroPat = pattern<1

    #compute one coil at a time to save working memory space
    for c in range(nCoils):
        coilPtr = coils[c,...]

        # todo: zeropad
        scratch = (coilPtr) * image
        scratch = ift2(scratch)

        # zero out non-sampled locations
        scratch[zeroPat]=0

        # ft back
        scratch = ft2(scratch)
        # todo: crop
        scratch = np.conj(coilPtr) * scratch

        # accumulate
        gradient = gradient + scratch[mask]


    gout = (gradient)
    gout.shape = (-1)
    return gout



