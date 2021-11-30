import numpy as np
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
    output = img*1
    phtons_total = N0*cellSize**2
    photons_nodule = phtons_total * noduleContrast/100
    
    size_celdas = np.ceil(noduleSize/cellSize)  # nº of cells occupied by the nodule

    # Position
    sy,sx=img.shape
    x = np.linspace(-sx/2, sx/2,sx)
    y = np.linspace(-sy/2, sy/2,sy)
    x, y = np.meshgrid(x, y) 
    d = np.sqrt(x**2 + y**2)
    
    output[d<=np.ceil(size_celdas/2)] = img[d<=np.ceil(size_celdas/2)] - photons_nodule
    return output

def plotMiddleLine(img, N0, cellSize):
    # The interval limits at 2*sigma where included in the plot
    sigma = np.std(img)
    mean = np.mean(img)
    filas, col = img.shape
    valores = img[filas//2,:]
    plt.plot(valores)
    plt.axhline(mean+2*sigma, color='black', ls='--')
    plt.axhline(mean-2*sigma, color='black', ls='--')
    plt.axhline(mean, color='black', ls='-')

def plotCellDistribution(img, numberOfBins):
    hist, bins= np.histogram(img, density=False, bins = numberOfBins)
    plt.plot(bins, np.insert(hist, 0, hist[0]), '-', drawstyle='steps',linewidth=1)
    plt.xlabel("No. photons")
    plt.ylabel("No. cells")
    return None