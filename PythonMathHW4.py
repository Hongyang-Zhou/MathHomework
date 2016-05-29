import numpy as np

dimension = 1

funcfamily = {
    "quad": {
        "f": lambda x: (x[0] + 2) * (x[0] + 2),
        "g": lambda x: np.array([2 * (x[0] + 2)]),
        "G": lambda x: np.array([2]),
        },
    "sin" : {
        "f": lambda x: np.sin(x[0]),
        "g": lambda x: np.array([np.cos(x[0])]),
        "G": lambda x: np.array([-np.sin(x[0])])
        }
    }


fname = "sin"
f = funcfamily[fname]["f"]
g = funcfamily[fname]["g"]
G = funcfamily[fname]["G"]

'''
dimension = 4


def f(x):
    return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4

def g(x):
    return np.array([
        2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3]) ** 3,
        20 * (x[0] + 10 * x[1]) + 4 * (x[1] - 2 * x[2]) ** 3,
        -8 * (x[1] - 2 * x[2]) ** 3 + 10 * (x[2] - x[3]),
        -40 * (x[0] - x[3]) ** 3 - 10 * (x[2] - x[3])])

def G(x):
    a11 = 2 + 120 * (x[0] - x[3]) ** 2
    a12 = 20
    a13 = 0
    a14 = -120 * (x[0] - x[3]) ** 2
    a21 = 20
    a22 = 200 + 12 * (x[1] - 2 * x[2]) ** 2
    a23 = -24 * (x[1] - 2 * x[2]) ** 2
    a24 = 0
    a31 = 0
    a32 = -24 * (x[1] - 2 * x[2]) ** 2
    a33 = 10 + 48 * (x[1] - 2 * x[2]) ** 2
    a34 = -10
    a41 = -120 * (x[0] - x[3]) ** 2
    a42 = 0
    a43 = -10
    a44 = 10 + 120 * (x[0] - x[3]) ** 2
    return np.array([[a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44]])
'''

find_minimum = True

def lmcalc():
    #x = [3, -1, 0, 1]
    x = [1]
    miu = 0.01
    epsilon = 0.0001
    k = 0
    lastf = 0
    lastq = 0
    while True:
        k += 1
        fval = f(x)
        gval = g(x)
        Gval = G(x)
        glen = np.sqrt(np.inner(gval, gval))
        if glen < epsilon:
            break
        while True:
            G2 = Gval + np.multiply(miu, np.identity(dimension))
            if np.all(np.linalg.eigvals(np.matrix(G2)) > 0):
                break
            miu *= 4
        s = np.linalg.solve(G2, np.multiply(-1, gval))
        fnext = f(x + s)
        gnext = g(x + s)
        Gnext = G(x + s)
        q = fval + np.inner(gval, s) + 0.5 * (np.matrix(s) * np.matrix(Gval) * np.matrix(s).transpose())
        qnext = fnext + np.inner(gnext, s) + 0.5 * (np.matrix(s) * np.matrix(Gnext) * np.matrix(s).transpose())
        if k > 1:
            r = (fnext - fval) / (qnext - q)
            #if find_minimum: r = -r
            if r < 0.25: miu *= 4
            elif r > 0.75: miu /= 2
            if r > 0: x += s
        lastq = q

        print "k = %s, x = %s" % (k, x)

    return x

print lmcalc()