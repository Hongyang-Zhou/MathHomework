import numpy
import random
import matplotlib.pyplot as plt

rmin = 0
rmax = 2 * numpy.pi
noise = 0.4
ns = [3, 5, 10]
ps = [10, 30, 100]


for nindex in range(0, len(ns)):
    for pindex in range(0, len(ps)):
        n = ns[nindex]
        p = ps[pindex]
        xs = []
        ys = []
        print p
        for ri in range(0, p):
            x = rmax / p * ri
            y = numpy.sin(x) + random.uniform(-0.5, 0.5) * noise
            xs.append(x)
            ys.append(y)

        if n >= p:
            continue
        plt.figure(nindex * len(ps) + pindex)
        plt.title("N: " + str(n) + "; Points: " + str(p))
        plt.plot(xs, ys, 'bo')
        coeffs = numpy.polyfit(xs, ys, n)
        xr = numpy.arange(rmin, rmax, .01)
        yr = numpy.polyval(coeffs, xr)
        sinline = plt.plot(xr, numpy.sin(xr), 'g--')
        polyline = plt.plot(xr, yr)
        plt.setp(polyline, color = 'r', linewidth=2.0)

plt.show()