import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import visualization_msgs.msg as visualization_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import pickle
import csv


from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.MotionPlanner import MotionPlanner
from exjobb.TA_PointClassification import PointClassification
from exjobb.NaiveRRTCPPAStar import NaiveRRTCPPAstar
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar import RandomBAstar
NUMBER_OF_START_POINTS = 10
start_points = np.array([ 
[-12.59000015,  11       ,    -5.29468489]   ,
[ 26.05999947, -11       ,   -10.37468719],
[  1.59000003, -12.5      ,    -5.66468811],
[16.5        , 8.69999981 , -5.3346858 ],
[-0.91000003 , 4         , -5.41468811],
[-20.28000069,   4.5      ,    -5.51468706],
[ 17.5       , -13.5      ,   -10.37468719],
[-10.84000015, -20.70000076,  -9.66468811],
[ 18.96999931, -11       ,    -5.75468397],
[ 23.05999947, -10.5      ,   -10.35468674]])

def fakeprint(object_to_print):
    return
    print(str(object_to_print))

def get_length_of_path(path):
    ''' Calculates length of the path in meters
    '''
    length = 0
    for point_idx in range(len(path) - 1):
        length += np.linalg.norm( path[point_idx] - path[point_idx + 1] )
    return length

def get_total_rotation(path):
    ''' Calculates the total rotation made by the robot while executing the path
    '''
    rotation = 0

    for point_idx in range(len(path) - 2):
        prev = (path[point_idx+1] - path[point_idx]) / np.linalg.norm( path[point_idx] - path[point_idx + 1])
        next = (path[point_idx+2] - path[point_idx+1]) / np.linalg.norm( path[point_idx+2] - path[point_idx + 1])
        dot_product = np.dot(prev, next)
        curr_rotation = np.arccos(dot_product)
        if not np.isnan(curr_rotation):
            rotation += abs(curr_rotation)

    return rotation



def main():
    with open('cached_coverable_points.dictionary', 'rb') as cached_pcd_file:
        cache_data = pickle.load(cached_pcd_file)
        coverable_points = cache_data["coverable_points"]
        traversable_points = cache_data["traversable_points"]
    
    traversable_pcd = PointCloud(fakeprint, points= traversable_points)
    
    motion_planner = MotionPlanner(fakeprint, traversable_pcd)

    def get_random_point():
        return traversable_points[np.random.randint(len(traversable_points))]

    def perform_cpp(cpp, start_point, start_point_nr):
        path = cpp.get_cpp_path(start_point)

        for sec in np.arange(0, 480, 10):
            #print("sec" + str(sec))
            minimal = min(cpp.data_over_time, key = lambda i: abs(i['time'] - sec))
            time = sec
            coveage = minimal["coverage"]
            length = get_length_of_path(path[0:minimal["path_point"]])
            rotation = get_total_rotation(path[0:minimal["path_point"]])
            stats = {
                "algorithm": cpp.name,
                "point": str(start_point_nr) + " - " + str(start_point),
                "time": time,
                "coverage": coveage,
                "length": round(length),
                "rotation": round(rotation)
            }
            results.append(stats)
            #stats = cpp.print_stats(path[0:minimal["path_point"]])
            #print(stats)
            #minimal = min(abs([x["time"] for x in cpp.data_over_time] - sec))
            #print(data)


        #stats = cpp.print_stats(path)
        #stats["Start point"] = str(start_point_nr) + " - " + str(start_point)
        
        print(stats["algorithm"] + " done.")
    
    results = []

    #for start_point_nr in range(NUMBER_OF_START_POINTS):
    #    start_point = get_random_point()

    for start_point_nr, start_point in enumerate(start_points[0:NUMBER_OF_START_POINTS]): 
        print("Start point " + str(start_point_nr) + ": " + str(start_point))
        #start_point = np.array(start_point)

        #cpp = NaiveRRTCPPAstar(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), 200)
        #perform_cpp(cpp, start_point, start_point_nr)

        cpp = Spiral(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), 480)
        perform_cpp(cpp, start_point, start_point_nr)

        cpp = BAstar(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), 480)
        perform_cpp(cpp, start_point, start_point_nr)

        cpp = BAstarVariant(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), 480)
        perform_cpp(cpp, start_point, start_point_nr)

        #cpp = RandomBAstar(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), 300)
        #perform_cpp(cpp, start_point, start_point_nr)

    with open('experiment_official_2.csv', 'w', newline='') as csvfile:
        fieldnames = ["point", 'algorithm', 'time', "coverage", "length", "rotation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

        

    


