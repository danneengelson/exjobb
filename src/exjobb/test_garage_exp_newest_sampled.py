import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import numpy as np
from copy import deepcopy
import pprint
import timeit
from exjobb.Environments import PointCloudEnvironment

from exjobb.PointCloud import PointCloud
from exjobb.MotionPlanner import MotionPlanner
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar3 import RandomBAstar3
from exjobb.Parameters import ROBOT_SIZE
from exjobb.full_test_ResultsShower import ResultShower
from exjobb.full_test_Experimenter import Experimenter
from exjobb.full_test_HyperOptimizer import HyptoOptimizer


###################

POINTCLOUD_FILE = 'garage.pcd'
TERRAIN_ASSESSMENT_FILE = 'garage_terrain_assessment.dictionary'
RESULTS_FILE = 'garage_newest_sampled.dictionary'
HYPER_MAX_EVAL = 100
NUMBER_OF_START_POINTS = 10
HYPER_START_POS = np.array([28.6, -6.7, -10.3])
start_points = {
    0: np.array([-12.59000015,  11.0,         -5.29468489]), 
    1: np.array( [26.05999947, -11.0,         -10.37468719]), 
    2: np.array( [1.59000003, -12.5 ,        -5.66468811]), 
    3: np.array([16.5   ,      8.69999981, -5.3346858 ]), 
    4: np.array([-0.91000003,  4.0     ,    -5.41468811]), 
    5: np.array( [-20.28000069,   4.5 ,        -5.51468706]), 
    6: np.array( [ 17.5 ,       -13.5,        -10.37468719]), 
    7: np.array( [-10.84000015, -20.70000076,  -9.66468811]), 
    8: np.array([ 18.96999931, -11.0,          -5.75468397]), 
    9: np.array( [ 23.05999947, -10.5,        -10.35468674])}


PRINT = True
ALGORITHMS = {}

###################

def my_print(text):
    if PRINT:
        return print(text)
    else:
        return 

def get_random_point(all_points):
    return all_points[np.random.randint(len(all_points))]

def save_data(data=None):
    with open(RESULTS_FILE, 'wb') as cached_pcd_file:
        if data is None:
            cache_data = deepcopy(ALGORITHMS)
        else:
            cache_data = deepcopy(data)
        for alg in cache_data.values():
            del alg["cpp"]
        pickle.dump(cache_data, cached_pcd_file) 

###################

def main():

    #with open(RESULTS_FILE, 'rb') as cached_pcd_file:
    #    cache_data = pickle.load(cached_pcd_file)
    #    pprint.pprint(cache_data)
    #return
    

    with open(RESULTS_FILE, 'rb') as cached_pcd_file:
        cache_data = pickle.load(cached_pcd_file)
        ALGORITHMS = deepcopy(cache_data)
        for alg in ALGORITHMS:
            ALGORITHMS[alg]["do_hyper"] = False
    
    
    

    #### STEP 1 - Get classified pointcloud ####

    environment = PointCloudEnvironment(my_print, TERRAIN_ASSESSMENT_FILE, POINTCLOUD_FILE)
    coverable_points = environment.coverable_pcd.points
    traversable_points = environment.traversable_pcd.points
    motion_planner = MotionPlanner(my_print, environment.traversable_pcd)
    
    #If from terrain assessment file:
    #with open(TERRAIN_ASSESSMENT_FILE, 'rb') as cached_pcd_file:
    #    cache_data = pickle.load(cached_pcd_file)
    #    coverable_points = cache_data["coverable_points"]
    #    traversable_points = cache_data["traversable_points"]
    #traversable_pcd = PointCloud(my_print, points= traversable_points)
    #motion_planner = MotionPlanner(my_print, traversable_pcd)

    #### STEP 2 - Hyper parameters ####
    for algorithm_key, algorithm in ALGORITHMS.items():
        if algorithm["do_hyper"]:
            trials = Trials()
            hyper_optimizer = HyptoOptimizer(save_data, algorithm, my_print, HYPER_START_POS, motion_planner, coverable_points)
            opt_param = fmin(   hyper_optimizer.hyper_test_newest_sampled_bastar_param,
                                space=( hp.uniform('ba_exploration', 0.75, 0.95), 
                                        hp.uniform('max_distance', 1, 5),  
                                        hp.uniform('max_distance_part_II', 4, 10),
                                        hp.uniform('min_bastar_cost_per_coverage', 5000, 10000), 
                                        hp.uniform('min_spiral_cost_per_coverage', 10000, 20000), 
                                        hp.uniform('step_size', 0.5, 1.0), 
                                        hp.uniform('visited_threshold', 0.25, 0.5)
                                    ),
                                algo=tpe.suggest,
                                max_evals=HYPER_MAX_EVAL,
                                trials=trials)
            print(trials.statuses())
            algorithm["opt_param"] = opt_param
            algorithm["hyper_data"] = trials.trials
            ALGORITHMS[algorithm_key] = algorithm
            save_data(ALGORITHMS)
 

    #### STEP 3 - Full tests ####
    for start_point_nr in range(NUMBER_OF_START_POINTS):
        #start_point = get_random_point(traversable_points)
        start_point = start_points[start_point_nr]
        print("Start point " + str(start_point_nr) + ": " + str(start_point))

        for algorithm_key, algorithm in ALGORITHMS.items():
            if algorithm["do_experiment"]:                
                experimenter = Experimenter(algorithm, print)
                parameters = None
                if "opt_param" in algorithm:
                    parameters = algorithm["opt_param"]
                    
                cpp = algorithm["cpp"](my_print, motion_planner, coverable_points, algorithm["experiment_time_limit"], parameters)

                if "sample_specific_stats" in algorithm:
                    experimenter.perform_sample_cpp(cpp, start_point, start_point_nr)
                    algorithm["sample_specific_stats"].append(experimenter.sample_specific_stats)
                else:
                    experimenter.perform_cpp(cpp, start_point, start_point_nr)

                algorithm["experiment_results"].append(experimenter.results)
                ALGORITHMS[algorithm_key] = algorithm
                save_data(ALGORITHMS)
    
    #### STEP 4 - Show results ####
    #shower = ResultShower(ALGORITHMS)
    #shower.show_coverage_per_time(400, 10)
    #shower.show_rotation_per_time(400, 10)
    #shower.show_cost_per_time(400, 10)
    #shower.show_length_per_time(400, 10)
    #shower.show_length_per_coverage(5)
    #shower.show_rotation_per_coverage(5)
    #shower.show_cost_per_coverage(5)
    #shower.show_hyper_parameter("step_size")
    
if __name__ == "__main__":
    main()
