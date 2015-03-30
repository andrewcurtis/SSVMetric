# Singular Value Metric Evaluator for K-Space Sampling Patterns in MRI

This is a reference implementation of the SSV metric. 
Two important classes are contained within. 

SamplingPattern defines default behaviour for all patterns, including their
simulation parameters (tolerance, iteration count, etc). There are two options
to set patterns, either set member variables _xvec_ and _yvec_ and call
calc\_pattern(), which sets the (2d) sampling pattern to the outer product of
the x and y patterns, mimicing the 'typical' case in 2d cartesian sampling.
Alternatively, one can simply set the member variable _sampling_ as a 2d numpy
array of the correct size, with entries of 1 at sampled loci and 0 indicating
locations that are not sampled.

The other class, PatternEvaluator handles evaluating the patterns including
addressing loading of coil sensitivities, masking, and interfacing with ARPACK.

Sample usage:

    import PatternEvaluator as pe
    import SamplingPattern as sp


    mype = pe.PatternEvaluator()
    mype.load_sens('../data/sens_6ch_128x128.npz')

    # make sampling pat
    mypat = sp.SamplingPattern()

    # set up vectors for r= 2x2
    mypat.yvec[::2,:]=0
    mypat.xvec[:,::2]=0
    # generate
    mypat.calc_pattern()

    #evaluate pattern
    mype.eval_pattern(mypat)


Note: Computation time can be significantly accelerated by replacing the
matrix-vector product system AtA in PatternEvaluator by a faster
implementation, in practice via CFFI leveraging FFTW. Due to portability
issues, we show the straight forward approach in pure python/numpy.



