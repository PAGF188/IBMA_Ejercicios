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
    print("coef_air:", coef_air)
    print("coef_water:", coef_water)
    print("coef_soft:", coef_soft)
    
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
    coef_air = round(read_file(energy, './coefs/coefAtenuacionAir.csv'),2)
    coef_adipose = round(read_file(energy, './coefs/coefAtenuacionAdipose.csv'),2)
    coef_breast = round(read_file(energy, './coefs/coefAtenuacionBreast.csv'),2)
    coef_soft = round(read_file(energy, './coefs/coefAtenuacionSoft.csv'),2)

    # offset added to leave some air space in one side
    offset = size//16

    # create 3 cubes
    # The size of the edge of each of these cubes will be half of the previous one.
    # biggest size: adipose tissue
    adipose_size = (size-offset)//2
    # medium size: breast tissue
    breast_size = adipose_size//2
    # smallest size: soft tissue
    soft_size = breast_size//2

    # phantom structure
    phantom = np.full((size,size,size), coef_air)

    # add adipose tissue
    # big left cube
    # adipose tissue coordinates
    at_x = size//4
    at_y = size//4
    at_z = 0
    #at_z = 0
    phantom[at_x:at_x+adipose_size, at_y:at_y+adipose_size, at_z:at_z+adipose_size+breast_size] = coef_adipose

    # small right cube
    # small adipose tissue coordinates
    sat_x = (3*size)//8
    sat_y = (3*size)//8
    sat_z = size//2 + size//8
    phantom[sat_x:sat_x+adipose_size//2, sat_y:sat_y+adipose_size//2, sat_z:sat_z+breast_size+offset] = coef_adipose

    # add breast tissue inside big left cube
    # breast tissue coordinates
    bt_x = (3*size)//8
    bt_y = (3*size)//8
    bt_z = size//8
    phantom[bt_x:bt_x+breast_size, bt_y:bt_y+breast_size, bt_z:bt_z+breast_size] = coef_breast

    # add soft tissue inside breast tissue
    # soft tissue inside breast tissue coordinates
    stbt_x = (7*size)//16
    stbt_y = (7*size)//16
    stbt_z = (3*size)//16
    phantom[stbt_x:stbt_x+soft_size, stbt_y:stbt_y+soft_size, stbt_z:stbt_z+soft_size] = coef_soft

    # add soft tissue in adipose tissue
    # soft tissue inside adipose tissue coordinates
    stat_x = (7*size)//16
    stat_y = (7*size)//16
    stat_z = size//2
    phantom[stat_x:stat_x+soft_size, stat_y:stat_y+soft_size, stat_z:stat_z+soft_size]= coef_soft

    # add soft tissue in the small right cube
    # soft tissue in small adipose tissue coordinates
    stsat_x = (7*size)//16
    stsat_y = (7*size)//16
    stsat_z = (12*size)//16
    phantom[stsat_x:stsat_x+soft_size, stsat_y:stsat_y+soft_size, stsat_z:stsat_z+soft_size]= coef_soft

    return phantom


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
