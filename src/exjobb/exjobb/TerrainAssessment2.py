import numpy as np
import open3d as o3d

# TUNING VALUES:
STEP_RESOLUTION = 3





class Pose:
    k_nearest_points = None
    center_point = None
    normal = None
    traversable = True 
    voxel_idx = None

class Position:
    def __init__(self, position):
        self.position = position

class TerrainAssessment2():

    def __init__(self, logger, pcd):
        self.logger = logger
        self.pcd = pcd.raw_pcd
        self.points = pcd.points
        self.pcd_kdtree = pcd.kdtree

    def analyse_terrain(self, start_pos):
        poses = []
        downpcd = self.pcd.voxel_down_sample(voxel_size=STEP_RESOLUTION)
        downpcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=1000))
        downpcd = downpcd.normalize_normals()
        #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=STEP_RESOLUTION)
        #for voxel in voxel_grid.get_voxels():
        #    self.logger.info(str(voxel.grid_index))
        plane_model, inliers = downpcd.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=1000)

        for idx, point in enumerate(np.asarray(downpcd.points)):
            new_pose = Pose()
            new_pose.k_nearest_points = [] #self.find_k_nearest(point, 4000)
            new_pose.center_point = point
            new_pose.normal = np.asarray(downpcd.normals[idx])
            if new_pose.normal[2] < 0:
                new_pose.normal = -new_pose.normal
            pitch = np.math.asin(new_pose.normal[2])
            #self.logger.info(str(pitch))
            if pitch < np.pi/4:
                new_pose.traversable = False
            
            #if idx in inliers:
            #    new_pose.traversable = False

            poses.append(new_pose)
        '''
        start_pos = np.array([9, 1, 2])
        
        visited = np.array([start_pos])
        queue = np.array([start_pos])
        valid = []

        while queue.size:
            self.logger.info("Queue: " + str(queue.shape))
            self.logger.info("Visited: " + str(queue.shape))
            pos, queue = queue[0, :], queue[1:,:]
            pose = self.get_pose_in_pos(pos)
            poses.append(pose)

            for neighbour in self.get_neighbour_pos(pos):
                if self.not_close_to_visited(neighbour, visited):
                    visited = np.append(visited, [neighbour], axis=0)
                    queue = np.append(queue, [neighbour], axis=0)
        '''
        o3d.visualization.draw_geometries([self.pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
        
        
        return poses



    def get_random_poses(self):
        poses = []

        while len(poses) < 100:
            random_idx = np.random.randint(0, len(self.points))
            pos = self.points[random_idx]
            pose = self.get_pose_in_pos(pos)
            poses.append(pose)

        return poses


    def get_neighbour_pos(self, pos):
        #neighbours = np.empty((1,3))
        resolution = STEP_RESOLUTION
        #self.logger.info(str(pos))
        nearest_point = self.find_k_nearest(pos, 1)[0]
        #OPTIMISE THIS
        while np.linalg.norm(pos - nearest_point) < 0.1:
            pos += np.array([0.0, 0.0, 0.05])
            nearest_point = self.find_k_nearest(pos, 1)[0]
            
        #self.logger.info(str(np.linalg.norm(pos - nearest_point)))
        #self.logger.info("nearest: " + str(nearest_point))
        #if np.linalg.norm(pos[0:2] - nearest_point[0:2]) > 0.5:
        #    return []
        
        positive_x = np.array([resolution, 0.0, 0.0])
        negative_x = np.array([-resolution, 0.0, 0.0])
        positive_y = np.array([0.0, resolution, 0.0])
        negative_y = np.array([0.0, -resolution, 0.0])
        neighbours = np.array([nearest_point + negative_y])
        for direction in [negative_x, positive_y, positive_x]:
            #self.logger.info(str(np.array([nearest_point + direction])))
            #self.logger.info(str(neighbours))
            #self.logger.info(str(np.array([nearest_point + direction]).shape))
            #self.logger.info(str(neighbours.shape))
            neighbours = np.append(neighbours, np.array([nearest_point + direction]), axis=0)
        #self.logger.info("neighbours: " + str(neighbours))
        #self.logger.info("neighbours shape;" + str(neighbours.shape))
        return neighbours
    
    def not_close_to_visited(self, pos, visited):
        for pose in visited:
            if np.linalg.norm(pos - pose) < 0.9*STEP_RESOLUTION:
                return False
        return True
    
    def get_pose_in_pos(self, start_pos):
        k = 500
        k_nearest = self.find_k_nearest(start_pos, k)
        center_point = np.average(k_nearest, axis=0)
        covariance_matrix = np.cov( k_nearest-center_point, rowvar=False )
        eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
        smallest_eig_val_idx = np.argmin(eig_vals)
        normal_vector = eig_vecs[:, smallest_eig_val_idx]
        if normal_vector[2] < 0:
            normal_vector = -normal_vector
        #self.logger.info(str(np.average(k_nearest, axis=0)))
        #self.logger.info(str(np.corrcoef( k_nearest-center_point, rowvar=False )))
        #self.logger.info("eig_vals " + str(eig_vals))
        #self.logger.info("eig_vecs " + str(eig_vecs))
        #self.logger.info("smallest_eig_val_idx " + str(smallest_eig_val_idx))
        #self.logger.info("normal_vector " + str(normal_vector))

        new_pose = Pose()
        new_pose.k_nearest_points = np.array(k_nearest)
        new_pose.center_point = center_point
        new_pose.normal = normal_vector/np.linalg.norm(normal_vector)
        pitch = np.math.asin(new_pose.normal[2])
        self.logger.info(str(pitch))
        if pitch < np.pi/4:
            new_pose.traversable = False

        return new_pose


    def find_k_nearest(self, point, k):
        #distances = np.linalg.norm(self.points - point, axis=1)
        #nearest_point_idx = np.argmin(distances)
        #sorted_points = self.points[np.argsort(distances)]
        [k, idx, _] = self.pcd_kdtree.search_knn_vector_3d(point, k)
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        return self.points[idx, :]

