import numpy as np
import timeit 
class Cell:
    def __init__(self, elevation):
        self.elevation = elevation

grid = np.empty((100, 100), dtype=object)
elevation = 0
with np.nditer(grid, flags=["refs_ok"], op_flags=['readwrite']) as it:
    for x in it:
        elevation += 2
        x[...] = Cell(elevation)

def get_item(x):
    return x.elevation

start_part = timeit.default_timer()
grid2 = np.vectorize(get_item)(grid)
end_part = timeit.default_timer()
total_part = end_part-start_part
print(grid2)
print(total_part)

start_part = timeit.default_timer()
with np.nditer(grid,flags=["refs_ok"], op_flags=['readonly']) as it:
    for x in it:
        grid2[...] = x.elevation
end_part = timeit.default_timer()
total_part = end_part-start_part
print(grid2)
print(total_part)