import numpy
import scipy.interpolate

def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return array[idx]

def func_inverse(x, y, val, kind='nearest', **kwargs):
    f = scipy.interpolate.interp1d(x, y, kind=kind, **kwargs)
    return f(val)
