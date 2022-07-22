#%%
import numpy as np
import pyvista
from utils_trimesh import mlab2pv,pv2mlab
import pymeshlab
from impalnt2mold import Reorient
import scipy.spatial

def generate_2d_contour(surf,resolution = 1000,s = [0.3,0.3]):
    surf_points_npy = 1.0*np.asarray(surf.points)
    mytree = scipy.spatial.cKDTree(surf_points_npy[:,:-1])
    surf_points_npy[:,-1] = 0
    surf_2d = pyvista.PolyData(surf_points_npy)
    x_min = np.min(surf_points_npy[:,0])
    x_max = np.max(surf_points_npy[:,0])
    y_min = np.min(surf_points_npy[:,1])
    y_max = np.max(surf_points_npy[:,1])
    box_x_max = (x_max + x_min)/2.0 + s[0] * (x_max - x_min)/2.0
    box_x_min = (x_max + x_min)/2.0 - s[0] * (x_max - x_min)/2.0
    box_y_max = (y_max + y_min)/2.0 + s[1] * (y_max - y_min)/2.0
    box_y_min = (y_max + y_min)/2.0 - s[1] * (y_max - y_min)/2.0

    x = np.linspace(box_x_min, box_x_max,resolution)
    y = np.linspace(box_y_min, box_y_max,int(resolution*(y_max-y_min)/(x_max-x_min)))
    lines1 = np.concatenate((np.ones_like(y)[:,None] * box_x_min, y[:,None]), axis=1)
    lines2 = np.concatenate((x[:,None] ,np.ones_like(x)[:,None] * box_y_max), axis=1)
    lines3 = np.concatenate((np.ones_like(y)[:,None] * box_x_max, y[::-1][:,None]), axis=1)
    lines4 = np.concatenate((x[::-1][:,None] ,np.ones_like(x)[:,None] * box_y_min), axis=1)
    lines = np.concatenate((lines1, lines2, lines3, lines4), axis = 0)

    
    # Recover the depth
    _, indexes = mytree.query([pt for pt in lines])
    depths = [surf.points[idx,-1] for idx in indexes ]
    lines = np.concatenate([lines, np.asarray(depths)[:,None]], axis = 1)
    return lines

target_path = '/home/sean/laser_ws/data/reg_data/surf_target.stl'
target_pv = pyvista.read(target_path)
pl = pyvista.Plotter()
pl.add_axes(line_width=5, labels_off=True)
pl.add_mesh(target_pv)

reorientor = Reorient()
target_pv, box = reorientor(target_pv)
pl.add_mesh(target_pv,'b')
lines = generate_2d_contour(target_pv)
line = pyvista.PolyData(lines)
pl.add_mesh(line)
pl.show()


# %%
