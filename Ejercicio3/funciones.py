import numpy as np
from scipy import interpolate
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
    coef_air = read_file(energy,'./coefs/coefAtenuacionAir.csv')           
    coef_water = read_file(energy,'./coefs/coefAtenuacionWater.csv')            
    coef_soft = read_file(energy,'./coefs/coefAtenuacionSoft.csv')   

    # prepare cube coordinates
    x, y, z = np.indices((size, size, size))
    
    cube1 = (x < size/2) & (y < size) & (z < size)
    cube2 = (x >= size/2) & (y < size) & (z < size)
    cube3 = ((x >= size/4) & (x<size-size/4)) & ((y >= size/4) & (y<size-size/4)) & ((z >= size/4) & (z<size-size/4))

    # combine the objects into a single boolean array
    phantom = np.full((size,size,size),coef_air)
    phantom[cube2] = coef_water
    phantom[cube3] = coef_soft

    return phantom

def breast_phantom(size,energy):
    return None

def vascular_phantom(size,energy):
    coef_soft = read_file(energy,'./coefs/coefAtenuacionSoft.csv')
    coef_blood = read_file(energy,'./coefs/coefAtenuacionBlood.csv')

    # Meshgrid para dibujar cilindro en 3D
    xx, yy, zz = np.mgrid[:size, :size, :size]
    circle = (xx - size//2) ** 2 + (yy - size//2) ** 2
    # Determinar su radio como el espacio que se quiere dejar en los bordes.
    # Since no dimensions are mentioned in statement, we assume:
    UMBRAL = 0.3
    RADIO_UMBRAL = circle[int(UMBRAL * size), int(UMBRAL * size),0] 

    # Ponemos los puntos del cilindro a "coef_blood" y el resto a "coef_soft"
    phantom = np.full((size,size,size),coef_soft)
    phantom[np.where(circle<=RADIO_UMBRAL)] = coef_blood

    return phantom
