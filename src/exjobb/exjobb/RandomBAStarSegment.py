from exjobb.Parameters import RANDOM_BASTAR_VARIANT_DISTANCE
import numpy as np
from exjobb.BAstar import BAstar

class BAStarSegment(BAstar):
    """A class to generate a segment of BAstar path. Used in Sample-Based BAstar CPP Algorithm.
    """
    def __init__(self, print, motion_planner, starting_point, angle_offset, visited_waypoints, coverable_pcd):
        """
        Args:
            print: function for printing messages
            motion_planner:  Motion Planner of the robot wihch also has the Point Cloud
            starting_point: A [x,y,z] np.array of the start position of the robot
            angle_offset: An angle in radians, representing the primary direction of the paths
            visited_waypoints: A Nx3 array with points that has been visited and should be avoided
        """
        super().__init__(print, motion_planner, coverable_pcd)


        self.start = starting_point
        self.path = visited_waypoints
        self.path = np.append(self.path, [starting_point], axis=0)    

        self.new_path = np.empty((0,3))
        next_starting_point = starting_point
        
        
        while True:
           
            path_to_cover_local_area, current_position = self.get_path_to_cover_local_area(next_starting_point, angle_offset)
            
            if len(path_to_cover_local_area) == 0:
                break


            self.follow_path(path_to_cover_local_area)   
            self.new_path = np.append(self.new_path, path_to_cover_local_area, axis=0)     


            next_starting_point = self.get_next_starting_point(self.path, angle_offset)   

            if next_starting_point is False:
                break

            distance_to_point = np.linalg.norm(next_starting_point - current_position)
            if distance_to_point < RANDOM_BASTAR_VARIANT_DISTANCE:
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            
            while path_to_next_starting_point is False:
                current_position = self.step_back()
                path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)

            self.follow_path(path_to_next_starting_point)
            self.new_path = np.append(self.new_path, path_to_next_starting_point, axis=0)  

        self.end = current_position
        self.covered_points_idx = self.coverable_pcd.covered_points_idx
        self.coverage = self.coverable_pcd.get_coverage_efficiency()
        self.path = self.new_path

        self.coverable_pcd = None
        self.traversable_pcd = None
        self.motion_planner = None
        self.print = None
