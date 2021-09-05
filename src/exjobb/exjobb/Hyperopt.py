from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import pickle
import open3d as o3d
import numpy as np
from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.MotionPlanner import MotionPlanner
from exjobb.TA_PointClassification import PointClassification
from exjobb.NaiveRRTCPPAStar import NaiveRRTCPPAstar
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar import RandomBAstar
from exjobb.Parameters import ROBOT_SIZE
BASTAR = 1
CURVED_BASTAR = 2
SPIRAL = 3
SAMPLED_BASTAR = 4


SMALL_POINT_CLOUD = True
ALGORITHM = BASTAR
TIME_LIMIT = 45


def fakeprint(object_to_print):
    return
    print(str(object_to_print))

def get_pcd():
    with open('cached_coverable_points.dictionary', 'rb') as cached_pcd_file:
        cache_data = pickle.load(cached_pcd_file)
        coverable_points = cache_data["coverable_points"]
        traversable_points = cache_data["traversable_points"]
    
    traversable_point_cloud = PointCloud(fakeprint, points= traversable_points)
    coverable_point_cloud = PointCloud(fakeprint, points= coverable_points)

    if SMALL_POINT_CLOUD:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            [-35, -35, -15.3],
            [3, 15, 10]
        )
        trav_points_idx = bbox.get_point_indices_within_bounding_box(traversable_point_cloud.raw_pcd.points)
        traversable_point_cloud = PointCloud(fakeprint, points= traversable_point_cloud.points[trav_points_idx])
        cov_points_idx = bbox.get_point_indices_within_bounding_box(coverable_point_cloud.raw_pcd.points)
        coverable_point_cloud = PointCloud(fakeprint, points= coverable_point_cloud.points[cov_points_idx])

    markers = []
    
    motion_planner = MotionPlanner(fakeprint, traversable_point_cloud)
    coverable_points = coverable_point_cloud.points

    return motion_planner, coverable_points


def get_bastar_path(args):
    step_size, visited_threshold = args
    cpp = BAstar(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), TIME_LIMIT)
    cpp.step_size = ROBOT_SIZE * step_size
    cpp.visited_threshold = visited_threshold * cpp.step_size
    print("cpp.step_size: " + str(cpp.step_size ))
    print("cpp.visited_threshold: " + str(cpp.visited_threshold ))
    path = cpp.get_cpp_path(start_point)
    return cpp

def get_curved_bastar_path(args):
    angle_offset, step_size, visited_threshold = args
    cpp = BAstarVariant(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), TIME_LIMIT)
    cpp.step_size = ROBOT_SIZE * step_size
    cpp.visited_threshold = visited_threshold * cpp.step_size
    path = cpp.get_cpp_path(start_point, angle_offset)
    return cpp

def get_spiral_path(args):
    step_size, visited_threshold = args
    cpp = Spiral(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), TIME_LIMIT)
    cpp.step_size = ROBOT_SIZE * step_size
    cpp.visited_threshold = visited_threshold * cpp.step_size
    path = cpp.get_cpp_path(start_point)
    return cpp

def get_sampled_bastar_path(args):
    step_size, visited_threshold  = args
    cpp = RandomBAstar(fakeprint, motion_planner, PointCloud(fakeprint, points= coverable_points), TIME_LIMIT)
    cpp.step_size = ROBOT_SIZE * step_size
    cpp.visited_threshold = visited_threshold * cpp.step_size
    path = cpp.get_cpp_path(start_point)
    return cpp

motion_planner, coverable_points = get_pcd()
start_point = [3, 15, -5.22789192199707]

def test(args):
    print(args)
    if ALGORITHM == BASTAR:
        cpp = get_bastar_path(args)
    elif ALGORITHM == CURVED_BASTAR:
        cpp = get_curved_bastar_path(args)
    elif ALGORITHM == SPIRAL:
        cpp = get_spiral_path(args)
    elif ALGORITHM == SAMPLED_BASTAR:
        cpp = get_sampled_bastar_path(args)

    stats = cpp.print_stats(cpp.path)
    print(stats)
    
    if stats["Coverage efficiency"] > 97:
        status = STATUS_OK
    else:
        status = STATUS_FAIL
    
    return {
        'loss': stats["Total rotation"] + stats["Length of path"],
        'status': status
    }

def main():
    
    trials = Trials()
    best = fmin(test,
        space=( hp.uniform('step_size', 0.3, 2), 
                hp.uniform('visited_threshold', 0.33, 1)),
                    
        algo=tpe.suggest,
        max_evals=2,
        trials=trials)
    print(best)
    trials_dict = trials.trials

    if ALGORITHM == BASTAR:
        cpp = get_bastar_path(best.values())
    elif ALGORITHM == CURVED_BASTAR:
        cpp = get_curved_bastar_path(best.values())
    elif ALGORITHM == SPIRAL:
        cpp = get_spiral_path(best.values())
    elif ALGORITHM == SAMPLED_BASTAR:
        cpp = get_sampled_bastar_path(best.values())

    with open('latest_path.dictionary', 'wb') as cached_pcd_file:
        cache_data = {
            "path": cpp.path, 
            "trials_dict": trials_dict
            }
        pickle.dump(cache_data, cached_pcd_file)