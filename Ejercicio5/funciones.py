import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise




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

def getNumberCellsPhoton(image, range_):
    hist, bins = np.histogram(image, density=False, bins = range_, range = (0, range_))
    return (bins, np.insert(hist, 0, hist[0]))

def getNumberPhotonsCell(image, range_):

    total = np.sum(image)
    hist, bins = np.histogram(image, density=False, bins=range_+1, range = (0, range_))
    hist = np.insert(hist, [hist.size], [0])
    hist = hist * bins
    
    # normalization (Because of the histogram computation of the bins is discrete
    # so we are loosing some precision:
    norm_factor = (total - np.sum(hist)) / hist.shape[0]
    hist = hist + norm_factor

    return (bins, hist)

def plotDistribution(data, xLabel, yLabel):
    plt.plot(data[0], data[1])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def plotLineH(qImage, pos):
    line = qImage[:,pos]
    plt.plot(line)
    plt.xlim(0, qImage.shape[0])
    plt.ylim(0, np.max(qImage)+5)
    plt.xlabel("X position")
    plt.ylabel("GL value")
    plt.show()

def detectorNoiseP(image, n1, n2):
    print("N Cells:", n1, n2)
    noisy = np.random.poisson(lam=image, size=(n1,n2))
    r = np.abs(image - noisy)
    image = image - r
    return image

def getContrast(image, fi1, col1, fi2, col2, w):
    valores = []
    for i in range(fi1, fi2, w):
        valores.append(np.mean(np.abs(image[i:i+w, col1] - image[i+w:i+2*w, col2]) / image[i:i+w, col1]))
    
    return np.max(valores)

def getSNR(image, a, b, w):
    media = np.mean(image[a-w:a+w, b-w:b+w])
    dev = np.std(image[a-w:a+w, b-w:b+w])
    return media/dev

 
    
