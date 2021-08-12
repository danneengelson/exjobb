####################
# Robot parameters #
####################

#Range from left to right
from numba.core.types.containers import SetEntry


ROBOT_SIZE = 0.75   

#Range from center to left/right
ROBOT_RADIUS = ROBOT_SIZE/2

ROBOT_HEIGHT = 1    

#When moving robot, update points around robot as covered every ROBOT_COVERAGE_STEP_SIZE meter
ROBOT_COVERAGE_STEP_SIZE = ROBOT_SIZE/2

#Maximum height difference between two points that the robot could manage 
ROBOT_STEP_SIZE = 0.25 * ROBOT_SIZE     

################################ 
#Terrain Assessment parameters #
################################

CELL_SIZE = 0.5
Z_RESOLUTION = 0.1
GROUND_OFFSET = 7*Z_RESOLUTION
MIN_FLOOR_HEIGHT = 2
MAX_STEP_HEIGHT = ROBOT_STEP_SIZE
MIN_POINTS_IN_CELL = CELL_SIZE**2 * 100
FLOOR_LEVEL_HEIGHT_THRESSHOLD = 100000 
MARGIN = 0.25

############################
#Motion Planner Parameters #
############################

STEP_SIZE = 0.1
RRT_STEP_SIZE = 3*STEP_SIZE
ASTAR_STEP_SIZE = 5*STEP_SIZE
UNTRAVERSABLE_THRESHHOLD = 2*STEP_SIZE
RRT_MAX_ITERATIONS = 10000

############################
#CPP Algorithms Parameters #
############################

# General
COVEREAGE_EFFICIENCY_GOAL = 0.4

# Naive RRT CPP
NAIVE_RRT_CPP_MAX_ITERATIONS = RRT_MAX_ITERATIONS
NAIVE_RRT_CPP_GOAL_CHECK_FREQUENCY = 50

# Spiral
SPIRAL_STEP_SIZE = 0.8 * ROBOT_SIZE
SPIRAL_VISITED_TRESHOLD = 0.66 * SPIRAL_STEP_SIZE

#Bastar
BASTAR_STEP_SIZE = 0.8 * ROBOT_SIZE
BASTAR_VISITED_TRESHOLD = 0.66 * BASTAR_STEP_SIZE

#Bastar variant
BASTAR_VARIANT_DISTANCE = 2

#Random BAstar
RANDOM_BASTAR_VISITED_TRESHOLD = 0.66 * BASTAR_STEP_SIZE
RANDOM_BASTAR_MAX_ITERATIONS = 70
RANDOM_BASTAR_NUMBER_OF_ANGLES = 1
RANDOM_BASTAR_PART_I_COVERAGE = 0.4
RANDOM_BASTAR_VARIANT_DISTANCE = 2   
RANDOM_BASTAR_MIN_COVERAGE = 0.01
RANDOM_BASTAR_MIN_SPIRAL_LENGTH = 3