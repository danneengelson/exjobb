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
from exjobb.RandomBAstar import RandomBAstar
from exjobb.Parameters import ROBOT_SIZE
from exjobb.full_test_ResultsShower import ResultShower
from exjobb.full_test_Experimenter import Experimenter
from exjobb.full_test_HyperOptimizer import HyptoOptimizer


###################

POINTCLOUD_FILE = 'bridge_2.pcd'
TERRAIN_ASSESSMENT_FILE = 'bridge_terrain_assessment.dictionary'
RESULTS_FILE = 'brdige_sampled.dictionary'
HYPER_MAX_EVAL = 100
NUMBER_OF_START_POINTS = 10
HYPER_START_POS = np.array([-53.7, 54.2, -2.7])
start_points = {
    0: np.array([-43.10443115,   3.99802136,   4.46702003]), 
    1: np.array([ 21.61431885, -33.00197983,  -2.77298403]), 
    2: np.array([-34.51068115,  12.49802208,  -4.17298126]), 
    3: np.array([ 15.9268198 , -36.00197983,  -2.6929822 ]), 
    4: np.array([38.98931885, 45.49802399,  1.19701743]), 
    5: np.array([ 3.73931861, 40.74802399,  2.83701849]), 
    6: np.array([ 15.5205698 , -31.50197792,  -2.8729825 ]), 
    7: np.array([-16.44818115, -19.25197792,  -3.58298159]), 
    8: np.array([10.52056885, 42.74802399,  2.46701956]), 
    9: np.array([53.89556885, 35.99802399,  0.33701676])}


PRINT = False
ALGORITHMS = {
    "Sampled BA*": {
        "name": "Sampled BA* & Inward Spiral",
        "do_hyper": True,
        "hyper_test": "sampled_bastar_param",
        "hyper_time_limit": 250,
        "hyper_min_coverage": 95,
        "do_experiment": True,
        "experiment_time_limit": 400,
        "experiment_results": [],
        "sample_specific_stats": [],
        "hyper_data": [],
        "formatted_hyper_data": [],
        "cpp": lambda print, motion_planner, cov_points, time_limit, parameters: RandomBAstar(print, motion_planner, PointCloud(print, points= cov_points), time_limit, parameters), 
        'line': 'm',
        'confidence_color': (1.0, 0.0, 1.0, 0.3)
    }
}

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
    

    #with open(RESULTS_FILE, 'rb') as cached_pcd_file:
    #    cache_data = pickle.load(cached_pcd_file)
    #    for alg in ALGORITHMS:
    #        if ALGORITHMS[alg]["do_hyper"]:
    #            ALGORITHMS[alg]["opt_param"] = cache_data[alg]["opt_param"]
    
    
    

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
            if algorithm_key == "BA*":
                opt_param = fmin(   hyper_optimizer.hyper_test_bastar,
                                    space=( hp.uniform('angle_offset', 0, np.pi*2),
                                            hp.uniform('step_size', 0.5, 1), 
                                            hp.uniform('visited_threshold', 0.5, 1)),
                                    algo=tpe.suggest,
                                    max_evals=HYPER_MAX_EVAL,
                                    trials=trials)
            elif algorithm_key == "Inward Spiral":
                opt_param = fmin(   hyper_optimizer.hyper_test_inward_spiral,
                                    space=( hp.uniform('step_size', 0.5, 1), 
                                            hp.uniform('visited_threshold', 0.5, 1)),
                                    algo=tpe.suggest,
                                    max_evals=HYPER_MAX_EVAL,
                                    trials=trials)
            elif algorithm_key == "Sampled BA*":
                coverage_2 = algorithm["hyper_min_coverage"]/100
                opt_param = fmin(   hyper_optimizer.hyper_test_sampled_bastar_param,
                                    space=( hp.uniform('coverage_1', 0.25, coverage_2), 
                                            hp.uniform('coverage_2', coverage_2-0.002, coverage_2), 
                                            hp.uniform('max_distance', 1, 10), 
                                            hp.uniform('max_distance_part_II', 4, 10),
                                            hp.uniform('max_iterations', 30, 150), 
                                            hp.uniform('min_bastar_coverage', 0.005, 0.05), 
                                            hp.uniform('min_spiral_length', 2, 100), 
                                            hp.uniform('nbr_of_angles', 7.6, 8.4),    
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
