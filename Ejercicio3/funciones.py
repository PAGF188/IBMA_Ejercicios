import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
from scipy import interpolate
import matplotlib.pyplot as plt
import os

archivos = ["./coefs/*"]

def read_file(energy_valor, path):

    d = {}
    with open(path, "r") as file:
        lines = file.readlines()[2:]
        for line in lines:
            aux = line.split("\t")
            # el archivo de blood tiene otro formato (1 celda separada por espacios)
            if os.path.basename(path) == "coefAtenuacionBlood.csv":
                aux = line.split(" ")
            d[float(aux[0])] = float(aux[1].split("\n")[0])
            
    if energy_valor in d.keys():
        return d[energy_valor]
    else:
        data = np.array(list(d.items()))
        f = interpolate.interp1d(data[:,0], data[:,1])
        return f(energy_valor)

def cube_phantom(size,energy):
    return None

def breast_phantom(size,energy):
    return None

def vascular_phantom(size,energy):
    coef_soft = read_file(energy,'./coefs/coefAtenuacionSoft.csv')
    coef_blood = read_file(energy,'./coefs/coefAtenuacionBlood.csv')

    phantom = np.full((size,size,size),coef_soft)
    
    return None

print(read_file(40,'./coefs/coefAtenuacionAir.csv'))
print(read_file(40,'./coefs/coefAtenuacionAdipose.csv'))
print(read_file(40,'./coefs/coefAtenuacionBreast.csv'))
print(read_file(40,'./coefs/coefAtenuacionSoft.csv'))
print(read_file(40,'./coefs/coefAtenuacionBlood.csv'))


vascular_phantom(256,40)