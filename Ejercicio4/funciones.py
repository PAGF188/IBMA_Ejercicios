import numpy as np
import pdb
import matplotlib.pyplot as plt


def createNoiseImageN(N0, n, cellSize):
    # Assuming standard deviation equal do poisson distribution.
    size = int(np.sqrt(n))
    image = np.random.normal(N0*cellSize**2,np.sqrt(N0*cellSize**2),size=(size,size))
    return image

def createNoiseImageP(N0, n, cellSize):
    size = int(np.sqrt(n))
    image = np.random.poisson(lam=N0*cellSize**2, size=(size,size))
    return image

def insertNodule(img, noduleSize, noduleContrast, N0, cellSize):
    return None

def plotMiddleLine(img, N0, cellSize):
    # The interval limits at 2*sigma where included in the plot
    sigma = np.std(img)
    mean = np.mean(img)
    filas, col = img.shape
    valores = img[filas//2,:]
    plt.plot(valores)
    plt.axhline(mean+2*sigma, color='black', ls='--')
    plt.axhline(mean-2*sigma, color='black', ls='--')
    plt.show()

def plotCellDistribution(img, numberOfBins):
    #plt.hist(img,numberOfBins, density=True, facecolor='b')
    
    hist, bins= np.histogram(img, density=False, bins = numberOfBins)
    plt.plot(bins, np.insert(hist, 0, hist[0]), '-', drawstyle='steps',linewidth=1)
    return None

N0= 4000000
n= 100*100   # Suppose a square shape detector
cellSize= 0.1
imgDataP= createNoiseImageP(N0, n, cellSize)
