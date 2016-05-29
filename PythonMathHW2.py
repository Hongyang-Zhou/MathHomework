import numpy
import matplotlib.pyplot as plt

trainpath = "C:/Users/windy/Documents/visual studio 2013/Projects/PythonMathHW2/PythonMathHW2/optdigits.tra"



datafile = open(trainpath)
traindata = datafile.read()
lines = traindata.split("\n")
vectors = []

for line in lines:
    tokens = line.split(",")
    for index in range(0, 64):
        tokens[index] = float(tokens[index])
    tokens[64] = int(tokens[64])
    vector = numpy.array(tokens[0:64])
    if tokens[64] == 3:
        vectors.append(vector)

    
vmatrix = numpy.array(vectors)
mean = vmatrix.mean(axis=0)
vmatrix = vmatrix - mean[numpy.newaxis, :]
vmatrix = vmatrix.transpose()
U, s, V = numpy.linalg.svd(vmatrix)

findex = 0

def plotxy(xdata, ydata):
    rawdata = U[0] * xdata + U[1] * ydata
    imgdata = []
    for i in range(0, 8):
        linedata = []
        for j in range(0, 8):
            c = (rawdata[i * 8 + j] / 16.0)
            linedata.append(c)
        imgdata.append(linedata)
    print imgdata
    fig, ax = plt.subplots()
    ax.imshow(imgdata, cmap=plt.get_cmap('hot') , interpolation = 'nearest')
    
    
    

coords = []
print s
for vector in vmatrix.transpose():
    x = vector.dot(U.transpose()[0])
    y = vector.dot(U.transpose()[1])
    coords.append([x, y])
    plt.plot(x, y, 'bo')

plt.show()
