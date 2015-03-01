import numpy as np
from numpy.fft import fft, ifft, fft2, fftshift, ifft2, ifftshift


# some utility functions to make running FFTs a little nicer.
# note that this uses fftpack, not FFTW..  in order to get fftw functionality,
# you have to compile numpy/scipy yourself _after_ installing ATLAS/BLAS
# and FFTW.

def ift(x, axis=[-1]):
    return ifftshift(ifft(ifftshift(x, axes=axis), axis=axis[0]), axes=axis)


def ft(x, axis=[-1]):
    return fftshift(fft(fftshift(x, axes=axis), axis=axis[0]), axes=axis)


def ift2(x):
    return (
        ifftshift(
            ifft2(ifftshift(x, axes=[-2, -1]), axes=[-2, -1]), axes=[-2, -1])
    )


def ft2(x):
    return (
        fftshift(
            fft2(fftshift(x, axes=[-2, -1]), axes=[-2, -1]), axes=[-2, -1])
    )


# square root Sum of squares over axis
def sumsq(x, axis=0):
    return np.sqrt(np.mean(abs(x) ** 2, axis=axis))




