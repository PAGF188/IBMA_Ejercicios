import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import interpolate
import os
import cv2
import pdb


# x-> ancho; y-> profundo; z->alto


####################################
# FUNCIONES PR8
####################################

def reconstructor(sinogram, nProj):

    
    nProjTotales = sinogram.shape[0]
    proj_size = sinogram.shape[1]

    # Calcular proyecciones a usar
    step = nProjTotales / nProj
    proyecciones = [int(i*step) for i in range(nProj)]

    # Calcular angulos de esas proyecciones
    step = 360/nProj
    angles = [i*step for i in range(nProj)]

    reconstruction = np.zeros((proj_size, proj_size))
    for pro, ang in zip(proyecciones, angles):
        temp = np.tile(sinogram[pro],(proj_size,1))
        temp = rotate(temp, ang)
        reconstruction += temp
    
    # Escalado entre 0 y 1:
    reconstruction = (reconstruction-np.min(reconstruction))/(np.max(reconstruction)-np.min(reconstruction))
    return reconstruction

def setHounsfield(Image, eE):
    agua = getCoef("./coefs/coefAtenuacionWater.csv", eE)
    print(f"Coef agua: {agua}")
    return 1000*(Image-agua)/(agua)

def displayWL(image, W, L, maxGL):
    
    output = image*1

    # Calcular limites ventana
    rang_min = L-W//2
    rang_max = L+W//2
    
    
    # Transformación lineal para el resto.
    # Based on: https://stackoverflow.com/questions/14224535/scaling-between-two-number-ranges
    output = (image - np.min(image)) * maxGL / (np.max(image)-np.min(image))

    # Eliminar elementos fuera de la ventana
    output[image<=rang_min] = 0
    output[image>=rang_max] = maxGL
    return output

##############################################################3
# RESTO DE FUNCIONES
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

def plotLineH(qImage, pos):
    line = qImage[:,pos]
    plt.plot(line)
    plt.xlim(0, qImage.shape[0])
    #plt.ylim(0, 1)
    plt.show()

def source(kVp, N0):
    eE = 0.4 * kVp
    return N0, eE

def interactor_CT(N0, Object, zpos, nProjections):

    step = 360/nProjections
    angles = [i*step for i in range(nProjections)]
    output = np.zeros((nProjections, Object.shape[0]))

    for n_pro, ang in enumerate(angles):
        # rotar
        aux = ndimage.rotate(Object, ang, axes=(0,1), reshape=False)
        for j in range(aux.shape[1]):
            output[n_pro][j] = N0
            for k in range(aux.shape[0]):
                output[n_pro][j] *= np.exp(-aux[k,j,zpos]/aux.shape[0])
    
    return output


def detectSinogram(qImage, nProjections, nDetectors):
    output = []

    for ang in range(nProjections):
        fila = detector_1D(qImage, ang, nDetectors)
        output.append(fila)
    
    return np.array(output)


def detector_1D(qImage, angle, nDetectors):
    pad = np.zeros((qImage.shape[0],nDetectors))
    aux = np.concatenate((pad, qImage, pad), axis=1)
    i_, j_ = aux.shape
    return aux[angle, j_//2 - nDetectors//2 : j_//2 + nDetectors//2 ]

def process_CT(image, n0):
    aux = np.divide(n0, image, out=np.zeros_like(image), where=image!=0)
    image = np.log(aux, where=0<(aux), out=np.zeros_like(image))
    return image
