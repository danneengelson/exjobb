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

FILE = 'garage_video.dictionary'
ALL_PATHS = [
    {
        "param" : {  'angle_offset': 3.44800051788481,
                        'step_size': 0.963400677899873,
                        'visited_threshold': 0.257015802906527}, 
        "cpp": "BA*",
        "ns": "bastar_1",
        "do_calc": True
    },
    {
        "param" : {  'angle_offset': 3.78341027362029,
                        'step_size': 0.601687134922371,
                        'visited_threshold': 0.328108983656107}, 
        "cpp": "BA*",
        "ns": "bastar_2",
        "do_calc": True
    },
    {
        "param" : {  'angle_offset': 5.27158130667689,
                        'step_size': 0.517468289229711,
                        'visited_threshold': 0.455659073558674}, 
        "cpp": "BA*",
        "ns": "bastar_3",
        "do_calc": True
    },
    {
        "param" : {  'angle_offset': 4.64664343656672,
                        'step_size': 0.633652049936913,
                        'visited_threshold': 0.472819723019576}, 
        "cpp": "BA*",
        "ns": "bastar_4",
        "do_calc": True
    },
    {
        "param" : {  'step_size': 0.999314930298507,
                        'visited_threshold': 0.32443603324225}, 
        "cpp": "Inward Spiral",
        "ns": "spiral_1",
        "do_calc": True
    },
    {
        "param" : {  'step_size': 0.825030992319859,
                        'visited_threshold': 0.433448258850281}, 
        "cpp": "Inward Spiral",
        "ns": "spiral_2",
        "do_calc": True
    },
    {
        "param" : {  'step_size': 0.521396930930628,
                        'visited_threshold': 0.47473068968531} , 
        "cpp": "Inward Spiral",
        "ns": "spiral_3",
        "do_calc": True
    },
    {
        "param" : {  'step_size': 0.627870706339337,
                        'visited_threshold': 0.498775709725593} , 
        "cpp": "Inward Spiral",
        "ns": "spiral_4",
        "do_calc": True
    },
    {
        "param" : {'ba_exploration': 0.90756041115558,
                                      'max_distance': 4.78202945337845,
                                      'max_distance_part_II': 6.75513650527977,
                                      'min_bastar_cost_per_coverage': 8192.530314616084,
                                      'min_spiral_cost_per_coverage': 12157.969167186768,
                                      'step_size': 0.562061544696692,
                                      'visited_threshold': 0.279490436505789}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_1",
        "do_calc": True
    },
    {
        "param" : {'ba_exploration': 0.816319265003861,
                                      'max_distance': 1.02476727664307,
                                      'max_distance_part_II': 4.76356301411862,
                                      'min_bastar_cost_per_coverage': 6616.530314616084,
                                      'min_spiral_cost_per_coverage': 19277.969167186768,
                                      'step_size': 0.950568870175564,
                                      'visited_threshold': 0.484179597225153}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_2",
        "do_calc": True
    },
    {
        "param" : {'ba_exploration': 0.853031300592955,
                                      'max_distance': 3.89663024793223,
                                      'max_distance_part_II': 4.80685526433465,
                                      'min_bastar_cost_per_coverage': 9312.530314616084,
                                      'min_spiral_cost_per_coverage': 13196.969167186768,
                                      'step_size': 0.636195719728099,
                                      'visited_threshold': 0.337665370485907}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_3",
        "do_calc": True
    },
    {
        "param" : {'ba_exploration': 0.8653615601139727,
                                      'max_distance': 4.129493635268686,
                                      'max_distance_part_II': 6.935911381739787,
                                      'min_bastar_cost_per_coverage': 8238.530314616084,
                                      'min_spiral_cost_per_coverage': 13644.969167186768,
                                      'step_size': 0.54868363557903,
                                      'visited_threshold': 0.3730115058138923}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_4",
        "do_calc": True
    },
    
]

PRINT = True

###################

def my_print(text):
    if PRINT:
        return print(text)
    else:
        return 

def save_data(data=None):
    with open(FILE, 'wb') as cached_pcd_file:
        if data is None:
            cache_data = deepcopy(ALL_PATHS)
        else:
            cache_data = deepcopy(data)

        pickle.dump(cache_data, cached_pcd_file) 

###################

def main():

    #### STEP 1 - Get classified pointcloud ####

    environment = PointCloudEnvironment(my_print, TERRAIN_ASSESSMENT_FILE, POINTCLOUD_FILE)
    coverable_points = environment.coverable_pcd.points
    traversable_points = environment.traversable_pcd.points
    motion_planner = MotionPlanner(my_print, environment.traversable_pcd)

    ###

    TIME_LIMIT = 400
    start_point = np.array([28.6, -6.7, -10.3])
    goal_coverage = 0.1
    paths_markers = []
    #Get CPP path
    for pub_path in ALL_PATHS:
        
        if not pub_path["do_calc"]:
            continue 

        coverable_pcd = PointCloud(my_print, points=coverable_points)
        if pub_path["cpp"] == "BA*":
            cpp = BAstar(my_print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
        if pub_path["cpp"] == "Inward Spiral":
            cpp = Spiral(my_print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
        if pub_path["cpp"] == "Sampled BA*":
            cpp = RandomBAstar3(my_print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
        
        ns = pub_path["ns"]
        pub_path["path"] = cpp.get_cpp_path(start_point, goal_coverage=goal_coverage)
        pub_path["markers"] = cpp.points_to_mark
        pub_path["stats"] = cpp.print_stats(pub_path["path"])
        save_data(ALL_PATHS)