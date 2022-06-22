from pickletools import read_unicodestring1
# from turtle import width
import pyvista
import numpy as np
from scipy import linalg as LA
from utils_trimesh import mlab2pv, split_surf,pv2mlab,\
    generate_cut_surf, pyvistaToTrimeshFaces, generate_cylinder,reduce_edge_distance, generate_top_surf_skirt
import pymeshlab

class Reorient(object):
    def __init__(self, *params):
        self.params = params
    
    def PCA(self, points):
        #centering the data
        shift = np.mean(points, axis = 0)
        points = points -  shift
        cov = np.cov(points, rowvar = False)
        evals , evecs = LA.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        return evecs

    def compute_rotMat(self, points):
        evecs = self.PCA(points)
        x,y,z = evecs[:,:3]
        self.R = np.asarray([x,y,z], np.double)

    def get_min_box(self, points, faces):
        points_b = np.matmul(np.transpose(self.R), np.transpose(points))
        points_cloud = pyvista.PolyData(np.transpose(points_b), faces=faces)
        box = pyvista.Cube(bounds=points_cloud.bounds)
        rotated_box = np.matmul(self.R, np.transpose(np.asarray(box.points)))
        min_box = np.transpose(rotated_box) 
        min_box = pyvista.PolyData(min_box, faces = box.faces)
        return min_box, box, points_cloud
    
    def transform(self, mesh):
        points = np.asarray(mesh.points)
        points = np.matmul(np.transpose(self.R), np.transpose(points))
        points = points.T
        points -= self.translation
        return pyvista.PolyData(points, faces = mesh.faces)

    def __call__(self,implant, s = (1.2, 1.2, 1.4)):
        # implant.save('orig_implant.ply')
        points = np.asarray(implant.points.copy())
        faces = implant.faces
        self.compute_rotMat(points)
        min_box, box, move_implant = self.get_min_box(points, faces)
        p = np.asarray(move_implant.points)
        self.translation = np.mean(p, axis = 0, keepdims=1)
        p -= self.translation
        implant = pyvista.PolyData(p, faces=implant.faces)
        implant.flip_normals()
        bounds = np.asarray(implant.bounds).reshape(3,2)
        if isinstance(s, tuple):
            s = np.asarray(s)
        bounds[:,0] = (np.asarray(implant.center) - (bounds[:,1] - bounds[:,0])/2.0 * s)
        bounds[:,1] = (np.asarray(implant.center) + (bounds[:,1] - bounds[:,0])/2.0 * s)
        box = pyvista.Cube(bounds=bounds.reshape(1,6)[0].tolist())
        # z_max = bounds.reshape(1,6)[0][-1]
        # bounds = np.asarray(implant.bounds).reshape(3,2)
        # bounds[:,0] = (np.asarray(implant.center) - (bounds[:,1] - bounds[:,0])/2.0 * 1.4)
        # bounds[:,1] = (np.asarray(implant.center) + (bounds[:,1] - bounds[:,0])/2.0 * 1.4)
        # big_box = pyvista.Cube(bounds=bounds.reshape(1,6)[0].tolist())

        return implant, box


if __name__ == '__main__':

    injection_hole_diameter = 10
    thickness = 5
    bottom_surf = pyvista.read('./data/reoriented_implant_bottom.stl')
    # top_origin_surf = pyvista.read('./data/reoriented_implant_top.stl')
    top_surf = generate_top_surf_skirt(bottom_surf, thickness)

    pl = pyvista.Plotter()
    

    implant = top_surf.merge(bottom_surf)
    reorientor = Reorient()
    implant, box = reorientor(implant)

    bottom_surf = reorientor.transform(bottom_surf)
    top_surf = reorientor.transform(top_surf)
    # pl.add_mesh(top_surf,'b')
    # pl.add_mesh(top_origin_surf)
    # pl.show()
 
    bottom_surf_points_npy = np.asarray(bottom_surf.points * 1.0)
    bottom_surf_points_npy[:,-1] = 0 # 
    bottom_surf_2d = pyvista.PolyData(bottom_surf_points_npy, bottom_surf.faces) 
    edges = bottom_surf_2d.extract_feature_edges()
    

    # pl.add_mesh(implant)
    

    #find the largest edge
    edge_points_bottom_2d = edges.connectivity(largest=True)
    largest_points = edge_points_bottom_2d.points[edge_points_bottom_2d.active_scalars==0]
    indexes = [bottom_surf_2d.find_closest_point(edge_pt) \
                for edge_pt in np.asarray(largest_points) ]
    edge_points = bottom_surf.points[indexes]


    cut_box = reduce_edge_distance(box, edge_points, s = 1.4)
    cut_box.flip_normals()


    cut_plane = generate_cut_surf(edge_points) # generate cut surface
    # pl.add_mesh(cut_plane, color='b')
    # pl.show()
    bottom_mold = cut_plane.merge(bottom_surf)
    bottom_mold = bottom_mold.extrude([0,0,70],capping=True)
    top_mold = cut_plane.merge(top_surf)
    top_mold = top_mold.extrude([0,0,-70],capping=True)
    bottom_mold.flip_normals() # Bug: the extruded mesh face normals are pointing inward
    top_mold.save('/tmp/top_mold.stl')
    bottom_mold.save('/tmp/bottom_mold.stl')
    cut_box.save('/tmp/cut_box.stl')
    # Convert to Meshlab
    ms = pymeshlab.MeshSet()
    # mesh[0]
    cutted_box_mlab = pymeshlab.Mesh(vertex_matrix=cut_box.points, 
                            face_list_of_indices=pyvistaToTrimeshFaces(cut_box.faces), 
                            v_normals_matrix=cut_box.point_normals, 
                            f_normals_matrix=cut_box.face_normals)
    ms.add_mesh(cutted_box_mlab)
    # mesh[1]
    top_mold_mlab = pymeshlab.Mesh(vertex_matrix=top_mold.points, face_list_of_indices=pyvistaToTrimeshFaces(top_mold.faces), v_normals_matrix=top_mold.point_normals, f_normals_matrix=top_mold.face_normals)
    # mesh[2]
    bottom_mold_mlab = pymeshlab.Mesh(vertex_matrix=bottom_mold.points, face_list_of_indices=pyvistaToTrimeshFaces(bottom_mold.faces), v_normals_matrix=bottom_mold.point_normals, f_normals_matrix=bottom_mold.face_normals)
 
    ms.add_mesh(top_mold_mlab)
    ms.add_mesh(bottom_mold_mlab)

    # Boolean operation using Meshlab
    # mesh[3]
    ms.mesh_boolean_difference(first_mesh=0, second_mesh=1)

    ms.save_current_mesh('mold_top.stl')
    # mesh[4]
    ms.mesh_boolean_difference(first_mesh=0, second_mesh=2)

    bottom_surf_lower = bottom_surf.extrude([0,0, -1 * injection_hole_diameter/2.0], capping = True)
    # bottom_surf_lower.flip_normals()
    bottom_surf_lower.compute_normals(inplace=True)
    bottom_surf_lower_mlab = pymeshlab.Mesh(vertex_matrix=bottom_surf_lower.points, 
                                face_list_of_indices=pyvistaToTrimeshFaces(bottom_surf_lower.faces), 
                                v_normals_matrix=bottom_surf_lower.point_normals, 
                                f_normals_matrix=bottom_surf_lower.face_normals)
    # mesh[5]
    ms.add_mesh(bottom_surf_lower_mlab)
    
    ms.mesh_boolean_difference(first_mesh = 4, second_mesh = 5)
    # pl.add_mesh(mlab2pv(ms[4]))
    mlab2pv(ms[4]).plot_normals()
    mlab2pv(ms[5]).plot_normals()
    pl.add_mesh(mlab2pv(ms[5]),color = 'b')
    pl.add_mesh(mlab2pv(ms[6]))
    pl.show()
    ms.save_current_mesh('mold_bottom.stl')
    

    # remove the end points and visualize
    # pl = pyvista.Plotter()
    # pl.add_mesh(mold)
    # pl.show()
