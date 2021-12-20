from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt




def source(kVp, N0):
    eE = 0.4 * kVp
    return N0, eE

def interactor_PR(N0, Object, Projection):
    if Projection == 'frontal':
        output = Object[Object.shape[0]//2,:,:] * 0
        for i in range(Object.shape[1]):
            for j in range(Object.shape[2]):
                output[i][j] = N0
                for k in range(Object.shape[0]):
                    output[i][j] *= np.exp(-Object[k,i,j]/Object.shape[0])
        return output

    elif Projection == 'lateral':
        output = Object[:, Object.shape[0]//2,:] * 0
        for i in range(Object.shape[0]):
            for j in range(Object.shape[2]):
                output[i][j] = N0
                for k in range(Object.shape[1]):
                    output[i][j] *= np.exp(-Object[i,k,j]/Object.shape[0])
        return output


def getNumberPhotons(image):
    return np.sum(image)

def getNumberCellsPhoton(image, range):
    hist, bins = np.histogram(image, density=False, bins = range, range = (0, range))
    return (bins, hist)

def getNumberPhotonsCell(image, range):
    # esta aqui
    hist, bins = np.histogram(image, density=False, bins = range, range = (0, range))
    # print(hist)
    # print(hist.shape)
    # print(bins[0:-1])
    # print(bins[0:-1].shape)

    return (bins, hist * bins[0:-1])

def plotDistribution(data, xLabel, yLabel):
    plt.plot(data[0], np.insert(data[1], 0, data[1][0]))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def plotLineH(qImage, pos):
    line = qImage[:,pos]
    plt.plot(line)
    plt.xlim(0, qImage.shape[0])
    plt.xlabel("X position")
    plt.ylabel("GL value")
    plt.show()

def detectorNoiseP(image, n1, n2):
    print("N Cells:", n1, n2)
    a = np.random.poisson(lam=0.3, size=(n1,n2))
    print(a)
    #print(image)
    return a
