import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import numpy as np
from copy import deepcopy
import pprint
import timeit


from exjobb.Parameters import ROBOT_SIZE

class HyptoOptimizer():

    def __init__(self, save, algorithm, print, hyper_start_pos, motion_planner, coverable_points):
        self.current_algorithm = algorithm
        self.hyper_start_pos = hyper_start_pos
        self.motion_planner = motion_planner
        self.coverable_points = coverable_points
        self.print = print
        self.save = save

    def get_random_angle(self):
        return np.pi*2 * np.random.randint(8) / 8

    def hyper_test(self, parameters):
        cpp = self.current_algorithm["cpp"](self.print, self.motion_planner, self.coverable_points, self.current_algorithm["hyper_time_limit"], parameters)
        path = cpp.get_cpp_path(self.hyper_start_pos, goal_coverage=self.current_algorithm["hyper_min_coverage"]/100)
        stats = cpp.print_stats(cpp.path)
        loss = stats["Total rotation"] + stats["Length of path"]
        self.print(stats)
        
        
        if stats["Coverage efficiency"] > self.current_algorithm["hyper_min_coverage"]:
            status = STATUS_OK
        else:
            status = STATUS_FAIL
        
        self.current_algorithm["formatted_hyper_data"].append({
            "parameters": parameters,
            "stats": stats,
            "status": status,
            "cost": loss
        })

        self.save()

        return {
            'loss': loss,
            'status': status,
            'stats': stats,
        }    

    def hyper_test_inward_spiral(self, args): 
        step_size, visited_threshold = args
        parameters = {
            "step_size":  ROBOT_SIZE * step_size,
            "visited_threshold": visited_threshold * ROBOT_SIZE * step_size
        }
        return self.hyper_test(parameters)

    def hyper_test_bastar(self, args):
        angle_offset, step_size, visited_threshold = args
        parameters = {
            "angle_offset": angle_offset,
            "step_size":  ROBOT_SIZE * step_size,
            "visited_threshold": visited_threshold * ROBOT_SIZE * step_size
        }
        
        
        return self.hyper_test(parameters)
        

    def hyper_test_sampled_bastar_param(self, args):
        coverage_1, coverage_2, max_distance, max_distance_part_II, max_iterations, min_bastar_coverage, min_spiral_length, nbr_of_angles, step_size, visited_threshold = args
        
        parameters = {
            "coverage_1": coverage_1,
            "coverage_2": coverage_2,
            "max_distance": max_distance,
            "max_iterations": max_iterations,
            "max_distance_part_II": max_distance_part_II,
            "min_spiral_length": min_spiral_length,
            "min_bastar_coverage": min_bastar_coverage,
            "nbr_of_angles": int(np.round(nbr_of_angles)),
            "step_size":  ROBOT_SIZE * step_size,
            "visited_threshold": visited_threshold * ROBOT_SIZE * step_size
        }        
        return self.hyper_test(parameters)