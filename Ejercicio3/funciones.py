import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
from scipy import interpolate
import matplotlib.pyplot as plt

archivos = ["./coefs/*"]

def read_file(energy_valor, path):

    d = {}
    with open(path, "r") as file:
        lines = file.readlines()[2:]
        for line in lines:
            aux = line.split("\t")
            d[float(aux[0])] = float(aux[1].split("\n")[0])
            
    if energy_valor in d.keys():
        return energy_valor * np.exp(-d[energy_valor])
    else:
        data = np.array(list(d.items()))
        aux1 = data[:,0]
        print(aux1.shape)
        aux2 = data[:,1]
        print(aux2.shape)
        
        f = interpolate.interp1d(data[:,0], data[:,1])
        ynew = f(energy_valor)   # use interpolation function returned by `interp1d`
        print(ynew)

        xnew = np.arange(1, 20000, 1)
        plt.plot(aux1, aux2, 'o', xnew, f(xnew), '-')
        plt.show()


def cube_phantom(size,energy):
    return None

def breast_phantom(size,energy):
    return None

def vascular_phantom(size,energy):
    return None

a = read_file(175,'./coefs/coefAtenuacionAdipose.csv')
print(a)