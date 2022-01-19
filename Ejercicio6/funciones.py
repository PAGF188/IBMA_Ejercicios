import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy import interpolate
import os
import pdb

def detector(image, n1, n2):
    # Suponiendo potencias 2
    x_orig, y_orig = image.shape

    factor_x = x_orig / n1
    factor_y = y_orig / n2
    print(factor_x, factor_y)

    nueva = np.zeros((n1,n2))

    if factor_x>=1 and factor_y>=1:
        factor_x = int(factor_x)
        factor_y = int(factor_y)
        for i,i_ in zip(range(0,n1), range(0,image.shape[0], factor_x)):
            for j,j_ in zip(range(0,n2), range(0,image.shape[1], factor_y)):
                nueva[i, j] = np.sum(image[i_:i_+factor_x, j_:j_+factor_y])
    else:
        M = np.matrix([[factor_x,0,0],[0,factor_y,0]])
        o_x, o_y = np.indices((n1, n2))  
        o_lin_homg_pts = np.stack((o_x.ravel(), o_y.ravel(), np.ones(o_y.size)))
        im_lin_pts = np.floor(M.dot(o_lin_homg_pts)).astype(int)
        nueva[o_lin_homg_pts[0].astype(int), o_lin_homg_pts[1].astype(int)] = image[im_lin_pts[0],im_lin_pts[1]]
    
    return nueva

def detectorNoiseFullP(image, n1, n2):
    # Suponiendo potencias 2
    x_orig, y_orig = image.shape

    factor_x = x_orig / n1
    factor_y = y_orig / n2
    print(factor_x, factor_y)

    nueva = np.zeros((n1,n2))

    if factor_x>=1 and factor_y>=1:
        factor_x = int(factor_x)
        factor_y = int(factor_y)
        for i,i_ in zip(range(0,n1), range(0,image.shape[0], factor_x)):
            for j,j_ in zip(range(0,n2), range(0,image.shape[1], factor_y)):
                nueva[i, j] = np.sum(image[i_:i_+factor_x, j_:j_+factor_y])
    else:
        M = np.matrix([[factor_x,0,0],[0,factor_y,0]])
        o_x, o_y = np.indices((n1, n2))  
        o_lin_homg_pts = np.stack((o_x.ravel(), o_y.ravel(), np.ones(o_y.size)))
        im_lin_pts = np.floor(M.dot(o_lin_homg_pts)).astype(int)
        nueva[o_lin_homg_pts[0].astype(int), o_lin_homg_pts[1].astype(int)] = image[im_lin_pts[0],im_lin_pts[1]]
        nueva = nueva * factor_x * factor_y

    for i in range(n1):
        for j in range(n2):
            ruido = np.random.poisson(lam=np.sqrt(nueva[i,j]))
            nueva[i,j] -=  ruido
    return nueva


def insertArtifact(obj, pos, size, mu):
    shape_ = obj.shape
    coords = np.ogrid[:shape_[0], :shape_[1], :shape_[2]]
    distance = np.sqrt((coords[0] - pos[0])**2 + (coords[1]-pos[1])**2 + (coords[2]-pos[2])**2) 
    circulo_coordenadas = (distance <= size/2)
    obj[circulo_coordenadas] = mu
    return obj

def getCoef(path, eE):
    d = {}
    with open(path, "r") as file:
        lines = file.readlines()[2:]
        for line in lines:
            aux = line.split("\t")
            # el archivo de blood tiene otro formato (1 celda separada por espacios)
            if os.path.basename(path) == "coefAtenuacionBlood.csv":
                aux = line.split(" ")
            if len(aux)==2:
                d[float(aux[0])] = float(aux[1].split("\n")[0])
            
    if eE in d.keys():
        return d[eE]
    else:
        data = np.array(list(d.items()))
        f = interpolate.interp1d(data[:,0], data[:,1])
        return f(eE)

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
    noisy = np.random.poisson(lam=0.1*image, size=(n1,n2))
    return image + noisy
    # noisy = np.random.poisson(lam=image, size=(n1,n2))
    # r = np.abs(image - noisy)
    # image = image - r
    # return image

def getContrast(image, fi1, col1, fi2, col2, w):
    valores = []
    for i in range(fi1, fi2, w):
        valores.append(np.mean(np.abs(image[i:i+w, col1] - image[i+w:i+2*w, col2]) / image[i:i+w, col1]))
    
    return np.max(valores)

def getSNR(image, a, b, w):
    media = np.mean(image[a-w:a+w, b-w:b+w])
    dev = np.std(image[a-w:a+w, b-w:b+w])
    return media/dev