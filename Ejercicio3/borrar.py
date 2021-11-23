from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
import pdb

def cubes(size):
    # prepare some coordinates
    x, y, z = np.indices((size, size, size))
    # draw cuboids in the top left and bottom right corners, and a link between
    # them
    cube1 = (x < Nx//2) & (y < Ny) & (z < Nz)
    cube2 = (x >= Nx//2) & (y < Ny) & (z < Nz)
    cube3 = (np.logical_and(x >= Nx/4, x<=Nx-Nx/4)) & (np.logical_and(y >= Ny/4,y<=Ny-Ny/4)) & (np.logical_and(z >= Nz/4,z<=Nz-Nz/4))    
    
    # combine the objects into a single boolean array
    phantom = np.full((Nx,Ny,Nz),1)
    phantom[cube2] = 2
    phantom[cube3] = 3

    pdb.set_trace()
    # # set the colors of each object
    # colors = np.empty(voxelarray.shape, dtype=object)
    # #colors[cube1] = 'blue'
    # #colors[cube2] = 'green'
    # colors[cube3] = 'red'
    # # and plot everything
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.voxels(voxelarray, facecolors=colors,  alpha=0.6)
    # #ax.axis('off')
    # plt.show()
size = 8
cubes(size)