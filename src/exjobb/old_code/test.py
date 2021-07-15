import numpy as np
import timeit 


start_part = timeit.default_timer()
hej = np.array([])

for i in range(100000):
    new_number = np.random.choice(10000, 1, replace=False)[0]
    hej = np.unique(np.append(hej, new_number))

#hej = np.unique(hej)

end_part = timeit.default_timer()
total_part = end_part-start_part
print(total_part)

