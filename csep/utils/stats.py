import numpy

def ecdf(x):
    xs = numpy.sort(x)
    ys = numpy.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
