####################
# Robot parameters #
####################

#Range from left to right
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