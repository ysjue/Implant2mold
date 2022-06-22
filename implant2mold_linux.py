from pickletools import read_unicodestring1
# from turtle import width
import pyvista
import numpy as np
from scipy import linalg as LA
from utils_trimesh import mlab2pv, split_surf,pv2mlab,generate_cut_surf, pyvistaToTrimeshFaces, generate_cylinder,reduce_edge_distance
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
    # implant = pyvista.read('./data/test.stl')
    # reorientor = Reorient()
    # implant, box = reorientor(implant)

    # bottom_surf = split_surf(implant, mode='bottom')  # find the top surface of implant
    # edges = bottom_surf.extract_feature_edges(30).plot()
    
    top_surf = pyvista.read('./data/reoriented_implant_top.stl')
    bottom_surf = pyvista.read('./data/reoriented_implant_bottom.stl')
    # edges = bottom_surf.extract_feature_edges(30)
    pl = pyvista.Plotter()
    # pl.add_mesh(top_surf)
    # pl.show()
    # exit()

    # pl.add_mesh(pyvista.PolyData(edges),color='blue')
    
    centroid= generate_cylinder(top_surf)
    implant = top_surf.merge(bottom_surf)
    reorientor = Reorient()
    implant, box = reorientor(implant)

    top_surf = reorientor.transform(top_surf)
    bottom_surf = reorientor.transform(bottom_surf)
    edges = bottom_surf.extract_feature_edges(30)

    pl.add_mesh(implant)
    
    centroid[0] += 6
    # inject_hole = pyvista.Cylinder(centroid, radius = 2.1, height = 22).triangulate()
    # inject_hole = pyvista.Cone(centroid, radius =6, direction = (-1,0,0),height = 30, resolution = 40).triangulate()
    length = 40
    width = 13
    height = 13
    x_min = centroid[0]-length/2.0
    x_max = centroid[0]+length/2.0
    y_min = centroid[1]-width/2.0
    y_max = centroid[1]+width/2.0
    z_min = implant.bounds[-2] - 0.2
    z_max = implant.bounds[-2] + height
    inject_hole = pyvista.Box((x_min,x_max,y_min,y_max,z_min,z_max)).triangulate()
    pl.add_mesh(inject_hole)

    meshset = pymeshlab.MeshSet()
    
    inject_hole = pymeshlab.Mesh(vertex_matrix=inject_hole.points, face_list_of_indices=pyvistaToTrimeshFaces(inject_hole.faces), v_normals_matrix=inject_hole.point_normals, f_normals_matrix=inject_hole.face_normals)
    meshset.add_mesh(inject_hole)

    implant = pymeshlab.Mesh(vertex_matrix=implant.points, face_list_of_indices=pyvistaToTrimeshFaces(implant.faces), v_normals_matrix=implant.point_normals, f_normals_matrix=implant.face_normals)
    meshset.add_mesh(implant)

    meshset.mesh_boolean_difference(first_mesh=0, second_mesh=1)
    # meshset.save_current_mesh('inject_hole.stl')
    # del meshset
    
    inject_hole = mlab2pv(meshset.current_mesh())
    inject_hole = inject_hole.connectivity(largest=True)
    inject_hole.plot()
    # idxes = np.arange(len(inject_hole.points))[inject_hole.active_scalars > 0]
    # inject_hole ,_= inject_hole.remove_points(idxes)
    # inject_hole = pyvista.PolyData(largest_points)
    # pl.add_mesh(top_surf)
    # pl.add_mesh(pyvista.PolyData(centroid),color='red')
    # pl.add_mesh(inject_hole)
    # pl.add_mesh(bottom_surf)
    # pl.show()



    #find the largest edge
    edge_points_bottom = edges.connectivity(largest=True)
    largest_points = edge_points_bottom.points[edge_points_bottom.active_scalars==0]

    cut_box = reduce_edge_distance(box, largest_points, s = 1.4)
    cut_box.flip_normals()
    pl.add_mesh(cut_box)
    pl.show()
    # pl.add_mesh(cutted_box)
    # pl.add_mesh(box)
    # pl.show()


    cut_plane = generate_cut_surf(largest_points) # generate cut surface
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
    # ms.load_new_mesh('./data/box.stl')
    cutted_box_mlab = pymeshlab.Mesh(vertex_matrix=cut_box.points, face_list_of_indices=pyvistaToTrimeshFaces(cut_box.faces), v_normals_matrix=cut_box.point_normals, f_normals_matrix=cut_box.face_normals)
    ms.add_mesh(cutted_box_mlab)
    # mesh[1]
    top_mold_mlab = pymeshlab.Mesh(vertex_matrix=top_mold.points, face_list_of_indices=pyvistaToTrimeshFaces(top_mold.faces), v_normals_matrix=top_mold.point_normals, f_normals_matrix=top_mold.face_normals)
    # mesh[2]
    bottom_mold_mlab = pymeshlab.Mesh(vertex_matrix=bottom_mold.points, face_list_of_indices=pyvistaToTrimeshFaces(bottom_mold.faces), v_normals_matrix=bottom_mold.point_normals, f_normals_matrix=bottom_mold.face_normals)
    # mesh[3]
    injection_hole_mlab = pymeshlab.Mesh(vertex_matrix=inject_hole.points, face_list_of_indices=pyvistaToTrimeshFaces(inject_hole.faces), v_normals_matrix=inject_hole.point_normals, f_normals_matrix=inject_hole.face_normals)
    ms.add_mesh(top_mold_mlab)
    ms.add_mesh(bottom_mold_mlab)
    ms.add_mesh(injection_hole_mlab)

    # Boolean operation using Meshlab
    ms.mesh_boolean_difference(first_mesh=0, second_mesh=1)
    ms.mesh_boolean_difference(first_mesh = 4, second_mesh = 3)
    ms.save_current_mesh('mold_top.stl')
    ms.mesh_boolean_difference(first_mesh=0, second_mesh=2)
    # ms.mesh_boolean_difference(first_mesh = 6, second_mesh = 3)
    ms.save_current_mesh('mold_bottom.stl')

    # remove the end points and visualize
    # pl = pyvista.Plotter()
    # pl.add_mesh(mold)
    # pl.show()
