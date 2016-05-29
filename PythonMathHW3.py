import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

np.random.seed(678)

def urd(max):
    return np.random.uniform() * 2 * max - max

def prd(max):
    return np.random.uniform() * max

def generate_random_variables(mean, cov, count):
    rd = np.random.multivariate_normal(mean, cov, count).tolist()
    return rd

dimension = 2
classes = 3

allpoints = []
allpoints += generate_random_variables([3, 3], [[1, 0], [0, 1]], 300)
allpoints += generate_random_variables([-3, 3], [[1, 0], [0, 1]], 300)
allpoints += generate_random_variables([3, -3], [[1, 0], [0, 1]], 300)

gaussians = []
pi = []

for i in range(0, classes):
    mean = [urd(3), urd(3)]
    cov = [[prd(2), 0], [0, prd(2)]]
    gaussian = st.multivariate_normal(mean, cov)
    gaussians.append(gaussian)
    pi.append(1.0 / classes)



iter_count = 30

w = []
for pindex in range(0, len(allpoints)):
    w.append([])
    for gindex in range(0, len(gaussians)):
        w[pindex].append(0)

for iter in range(0, iter_count):
    
    for pindex in range(0, len(allpoints)):
        wsum = 0
        for gindex in range(0, len(gaussians)):
            pdf = gaussians[gindex].pdf(allpoints[pindex]) * pi[gindex]
            w[pindex][gindex] = pdf
            wsum += pdf
        for gindex in range(0, len(gaussians)):
            w[pindex][gindex] /= wsum

    wsums = np.sum(w, 0)

    for gindex in range(0, len(gaussians)):
        pi[gindex] = wsums[gindex] / len(allpoints);
        
        mean = [0] * dimension
        for pindex in range(0, len(allpoints)):
            mean += np.multiply(w[pindex][gindex], allpoints[pindex])
        mean /= wsums[gindex]
        gaussians[gindex].mean = mean

        cov = [[0] * dimension] * dimension
        for pindex in range(0, len(allpoints)):
            dist = allpoints[pindex] - mean
            cov += np.multiply(w[pindex][gindex], np.outer(dist, dist))
        cov /= wsums[gindex]
        gaussians[gindex].cov = cov


    

maxs = np.max(w, 1)
for pindex in range(0, len(allpoints)):
    if maxs[pindex] == w[pindex][0]: style = 'ro'
    elif maxs[pindex] == w[pindex][1]: style = 'go'
    else: style = 'bo'
    item = allpoints[pindex]
    plt.plot(item[0], item[1], style)
plt.show()