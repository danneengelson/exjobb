from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.Floor import Floor
from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator
import matplotlib.pyplot as plt
from polylidarutil import (generate_test_points, plot_points, plot_triangles, get_estimated_lmax,
                            plot_triangle_meshes, get_triangles_from_he, get_plane_triangles, plot_polygons)

RRT_STEP_SIZE = 3*STEP_SIZE
COVEREAGE_EFFICIENCY_GOAL = 0.8
MAX_ITERATIONS = 10000
GOAL_CHECK_FREQUENCY = 50

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class BoustrophedonCPP(CPPSolver):

    def __init__(self, logger, ground_pcd):
        self.name = "Boustrophedon Cell Decomposition"
        super().__init__(logger, ground_pcd)

    def get_cpp_path(self, start_point):
        self.start_tracking()
        ground_hegihts = [-11.04, -5.98, 2.29]
        for idx, ground_hegiht in enumerate(ground_hegihts[:-1]):
            heights = np.asarray(self.pcd.points)[:,2] 
            points_idx = np.where( np.logical_and(heights > ground_hegiht, heights < ground_hegihts[idx+1] ))[0]
            
            
            random = np.random.randint(len(points_idx), size=100)
            path = np.asarray(self.pcd.points)[points_idx][:,0:2]
            
            kwargs = dict(alpha=0.0, lmax=1.0)
            lmax = get_estimated_lmax(**kwargs)
            polygons = extractPolygons(path, alpha=1.0, lmax=1, minTriangles=5)
            delaunay = Delaunator(path)
            self.print(polygons)


            fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
            # plot points
            plot_points(path, ax)
            # plot all triangles
            #plot_triangles(get_triangles_from_he(delaunay.triangles, path), ax)
            # plot mesh triangles
            #triangle_meshes = get_plane_triangles(planes, delaunay.triangles, path)
            #plot_triangle_meshes(triangle_meshes, ax)
            # plot polygons
            plot_polygons(polygons,delaunay, points=path, ax=ax)

            plt.axis('equal')

            plt.show()

            self.print(len(points_idx))
            
            self.print(path)
            return path

        #tree = self.build_RRT_tree(start_point)
        path = []
        self.print_stats(path)
        return path

