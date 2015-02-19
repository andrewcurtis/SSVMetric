import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError

import time

# local package imports
from SamplingPattern import SamplingPattern
from defaults import base_sz
from utils import ft, ift, ft2, ift2
from PatternEvaluator import PatternEvaluator



class R2StarEvaluator(PatternEvaluator):
    """
    This class overrides system matrix in patternEvaluator to handle 
    multi-echo experiments. The generic multi echo case with no signal relaxation
    or dephasing isn't interesting (it is the same as a single echo with more samples).
    However, given a parametric map (R2 star) in this case, the echo sampling order
    is important. 
    
    Use this as a template for other similar signal models. (T1, T2, FLAIR, etc).
    
    Here we specialize for the case of parametric mapping of the R2 star values.
    """
    
    def __init__(self, base_sz = BASE_N, sens=[], max_tries=2, n_echoes=1):
        super(R2StarEvaluator, self).__init__(base_sz, sens, max_tries)
        
        self.n_echoes = n_echoes
        
        self.r2star = np.zeros((self.base_sz, self.base_sz), dtype='complex')
        self.s0 = np.zeros((self.base_sz, self.base_sz), dtype='complex')
    
        self.sampling = np.zeros((self.n_echoes, self.base_sz, self.base_sz), dtype='float')
        
    
    def calc_AtA(self):
        """ 
        calculate system matrix (normal equations) 
        """
        nSamp = np.sum(sampling)
        maskSz = np.sum(mask)
        nCoils, nv, npts = self.sens.shape
        if x0.dtype <> np.complex128:
            x0 = x0.astype('complex128')
        x_img = x0[self.mask]

        A_fwd = np.zeros(nSamp*nCoils, dtype='complex')

        result = np.zeros(maskSz, dtype='complex')

        # Compute A
        sys_fwd_r2star(x_img, A_fwd, sens, sampling>0, mask)

        #compute A^H
        A_back = sys_bck_r2star(A_fwd, sens, sampling>0, mask)
        
        result[:] = A_back[:] #copy / flatten

        return result
   
  
def sys_fwd_r2star(im_mask, data, coils, pattern, mask, s0, r2):
    """
    linear system to go from image --> kspace
    """
    nCoils, nv, npts  = coils.shape
    ne = pattern.shape[0]
    #print coils.shape
    #print data.shape
    image = np.zeros((nv, npts), dtype='complex128')
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
def sys_bck_r2star(data, coils, pattern, mask, s0, r2):
    """
    linear system to go from kspace --> image
    """
    nCoils, nv, npts  = coils.shape
    #print coils.shape
    #print data.shape
    data.shape= (nCoils,-1)
    nD = coils[0].ndim
    accum = 0.0
    tmpGrad = []
    maskSz = np.sum(mask)
    gradient = np.zeros(maskSz)
    scratch = np.zeros_like(coils[0])

    #for each coil, do S^H . I
    for c in range(nCoils):
        # slice out current coil
        coilPtr = coils[c,...]
        dataPtr = data[c,...]

        scratch = scratch*0

        scratch[pattern] = dataPtr[:]

        scratch = ft2(scratch)
        # todo: crop
        scratch = np.conj(coilPtr) * scratch

        # accumulate
        gradient = gradient + scratch[mask]
        #print np.sum(np.abs(scratch.ravel()))


    data.shape = (-1)
    gout = (gradient)
    gout.shape = (-1)
    return gout 