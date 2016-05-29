import math
import numpy as np
import numpy.matlib as mat

import matplotlib.pyplot as plt


def qsubp(H, c, Ae, be):
    if len(Ae) == 0:
        xz = np.linalg.pinv(H) * -c
        l = None
        return xz, l
    else:
        MM = np.hstack((H, Ae.transpose()))
        MM = np.vstack((MM, np.hstack((Ae, mat.zeros((Ae.shape[0], MM.shape[1] - Ae.shape[1]))))))
        xvec = np.linalg.pinv(MM) * np.vstack((-c, -be))
        x = xvec[:H.shape[0]]
        l = -xvec[H.shape[0]:]
        return x, l

def QuadProg(H, c, Ae, be, Ai, bi, x0):
    epsilon = 1e-9
    err = 1e-6
    k = 0
    x = x0
    n = len(x)
    kmax = 1e2
    ne = len(be)
    ni = len(bi)
    index = mat.ones((ni, 1))
    for i in range(0, ni):
        if Ai[i,:] * x > bi[i] + epsilon:
            index[i] = 0

    while k <= kmax:
        Aee = []
        #if ne > 0:
        #    Aee = Ae
        AeeList = []
        if ne > 0:
            AeeList.append(Ae)
        for j in range(0, ni):
            if index[j] > 0:
                AeeList.append(Ai[j, :])
                #Aee = mat.vstack((Aee, Ai[j,:]))
        if len(AeeList) == 0:
            Aee = []
        else:
            Aee = mat.vstack(AeeList)
            m1, n1 = mat.shape(Aee)
        gk = H * x + c
        dk, lamk = qsubp(H, gk, Aee, mat.zeros((m1, 1)))
        if np.linalg.norm(dk) <= err:
            y = 0.0
            if len(lamk) > ne:
                jk = np.argmin(lamk[ne:len(lamk)])
                y = lamk[jk]
            if y >= 0:
                exitflag = 0
            else:
                exitflag = 1
                for i in range(0, ni):
                    if index[i] and (ne + sum(index[0:i]))  == jk:
                        index[i] = 0;
                        break;
            #k += 1
        else:
            exitflag = 1
            alpha = 1.0
            tm = 1.0
            for i in range(0, ni):
                if index[i] == 0 and Ai[i, :] * dk < 0:
                    tmm = (bi[i] - Ai[i, :] * x) / (Ai[i, :] * dk)
                    tm1 = abs(tmm[0,0])
                    if tm1 < tm:
                        tm = tm1;
                        ti = i;
            alpha = min(alpha, abs(tm))
            x = x + alpha * dk
            if tm < 1:
                index[ti] = 1
        if exitflag == 0:
            break
        k += 1
    print k
    return x


def svmfit(X, y):
    features, samples = np.shape(X)
    H = np.ones((features + 1, features + 1))
    H[0] = 0
    H[:, 0] = 0
    A = np.zeros((samples, features + 1))
    for i in range(samples):
        A[i] = np.hstack((y[i], y[i] * X[:, i]))
    w = mat.ones(features + 1)
    w = QuadProg(np.matrix(H), mat.zeros((features + 1, 1)), [], [], np.matrix(A), mat.ones((samples, 1)), np.matrix(w).transpose());
    return w

def svmpredict(w, x):
    
    return np.sign(w.transpose() * x);

traindata = []
trainflag = []
r = 0.05
for i in range(20):
    x1 = np.random.rand()
    x2 = np.random.rand()
    traindata.append([x1, x2])
    if x1 > x2:
        x1 += r
        x2 -= r
        trainflag.append(1)
    else:
        x1 -= r
        x2 += r
        trainflag.append(-1)
traindata = np.array(traindata).transpose()

testdata = []
for i in range(100):
    x1 = np.random.rand()
    x2 = np.random.rand()
    if x1 > x2:
        x1 += r
        x2 -= r
    else:
        x1 -= r
        x2 += r
    testdata.append([x1, x2])
testdata = np.array(testdata).transpose()

svmW = svmfit(traindata, trainflag)
testflag = svmpredict(svmW, testdata)
