import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import os
import pdb

# x-> ancho; y-> profundo; z->alto


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
    _, j_ = qImage.shape
    return qImage[angle, j_//2 - nDetectors//2 : j_//2 + nDetectors//2 ]

def process_CT(image, n0):
    image = np.log(n0/image)
    return image
