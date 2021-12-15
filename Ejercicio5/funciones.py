import numpy as np
from scipy import interpolate
import os


def source(kVp, N0):
    eE = 0.4 * kVp
    return N0, eE

def interactor_PR(N0, Object, Projection):
    if Projection == 'frontal':
        output = Object[Object.shape[0]//2,:,:] * 0
        for i in range(Object.shape[1]):
            for j in range(Object.shape[2]):
                valor = 0
                for k in range(Object.shape[0]):
                    valor += N0 * np.exp(-Object[k,i,j])
                output[i][j] = valor
        return output

    elif Projection == 'lateral':
        pro = Object[Object.shape[0]//2,:,:]
        return pro * N0

def plotLineH(qImage, pos):
    line = qImage[:,pos]
    return line
