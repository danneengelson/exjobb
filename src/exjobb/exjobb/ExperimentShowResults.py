
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt



def get_average(data, algorithm, property, time):

    rows = list(filter(lambda row: row['algorithm'] == algorithm and int(row['time']) == time, data))
    
    values = [float(row[property]) for row in rows]
    return np.mean(values)

def get_average_for_coverage(data, algorithm, property, coverage):
    
    points = set( [row['point'] for row in data] )
    rows = []
    for point in points:
        rows_for_point = list(filter(lambda row: row['algorithm'] == algorithm and row['point'] == point, data))
        min_idx = np.argmin(np.array([abs(float(row['coverage']) - coverage) for row in rows_for_point]) )
        rows.append(rows_for_point[min_idx])

    values = [float(row[property]) for row in rows]
    return np.mean(values)

def get_average_for_max_dist(data, algorithm, property, max_dist):
    
    rows_for_max_dist = list(filter(lambda row: row['algorithm'] == algorithm and float(row['Max dist']) == max_dist, data))
    values = [float(row[property]) for row in rows_for_max_dist]
    return np.mean(values)

def get_error_for_max_dist(data, algorithm, property, max_dist):
    
    rows_for_max_dist = list(filter(lambda row: row['algorithm'] == algorithm and float(row['Max dist']) == max_dist, data))
    values = [float(row[property]) for row in rows_for_max_dist]
    return np.std(values)

def get_error(data, algorithm, property, time):
    #print(data)
    #rows2 = list(filter(lambda row:  int(row["time"]) == time, data))

    rows = list(filter(lambda row: row['algorithm'] == algorithm and int(row['time']) == time, data))
    
    #print(rows)
    values = [float(row[property]) for row in rows]
    return np.std(values)

def get_error_for_coverage(data, algorithm, property, coverage):
    points = set( [row['point'] for row in data] )
    #print(points)
    rows = []
    for point in points:
        rows_for_point = list(filter(lambda row: row['algorithm'] == algorithm and row['point'] == point, data))
        #print(rows_for_point)
        #print(np.array([float(row['coverage']) - coverage for row in rows_for_point]))
        min_idx = np.argmin(np.array([abs(float(row['coverage']) - coverage) for row in rows_for_point]) )
        rows.append(rows_for_point[min_idx])

    values = [float(row[property]) for row in rows]
    return np.std(values)


def get_random_bastar_coverage(times, data):
    average_coverage = 100*np.array([get_average(data, "Random BAstar", "coverage", time) for time in times])        
    error = np.array([get_error(data, "Random BAstar", "coverage", time) for time in times])   
    plt.plot(times, average_coverage)
    #plt.fill_between(times, average_coverage-error, average_coverage+error, color=colors[alg_idx])

    plt.ylabel('Coverage [%]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Coverage over Time')
    plt.xlim(0,300)
    plt.ylim(0,105)

    plt.show()


algorithms = ["Inward Spiral", "BAstar", "BAstar Variant"]
line = ["b", "r:", "g--"]
labels = ["Inward Spiral", "BA*", "Curved BA*"]
colors = [(0.0,0.0,1.0,0.3), (1.0,0.0,0.0,0.3), (0,1.0,0,0.3)]  

def get_coverage(times, data):
    for alg_idx, algorithm in enumerate(algorithms):
        average_coverage = np.array([get_average(data, algorithm, "coverage", time) for time in times])        
        error = np.array([get_error(data, algorithm, "coverage", time) for time in times])   
        plt.plot(times, average_coverage, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(times, average_coverage-error, average_coverage+error, color=colors[alg_idx])

    plt.ylabel('Coverage [%]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Coverage over Time')
    plt.xlim(0,300)
    plt.ylim(0,105)

    plt.show()

def get_length(times, data):
    for alg_idx, algorithm in enumerate(algorithms):
        
        average_length = np.array([get_average(data, algorithm, "length", time) for time in times])        
        if not len(average_length):
            continue
        error = np.array([get_error(data, algorithm, "length", time) for time in times])   
        plt.plot(times, average_length, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(times, average_length-error, average_length+error, color=colors[alg_idx])

    plt.ylabel('Length [m]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Length of Path over Time')
    plt.xlim(0,300)
    #plt.ylim(0,100)

    plt.show()

def get_rotation(times, data):
    for alg_idx, algorithm in enumerate(algorithms):
        
        average_rotation = np.array([get_average(data, algorithm, "rotation", time) for time in times])        
        if not len(average_rotation):
            continue
        error = np.array([get_error(data, algorithm, "rotation", time) for time in times])   
        plt.plot(times, average_rotation, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(times, average_rotation-error, average_rotation+error, color=colors[alg_idx])

    plt.ylabel('Rotation [rad]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Total Rotation over Time')
    plt.xlim(0,480)
    #plt.ylim(0,100)

    plt.show()

def get_length_per_coverage(coverages, data):
    for alg_idx, algorithm in enumerate(algorithms):
        
        average_length = np.array([get_average_for_coverage(data, algorithm, "length", coverage) for coverage in coverages])        
        if not len(average_length):
            continue
        error = np.array([get_error_for_coverage(data, algorithm, "length", coverage) for coverage in coverages])   
        plt.plot(coverages, average_length, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(coverages, average_length-error, average_length+error, color=colors[alg_idx])

    plt.ylabel('Length [m]')
    plt.xlabel('Coverage [%]')
    plt.legend()
    plt.title('Length of Path per Coverage')
    #plt.xlim(0,300)
    #plt.ylim(0,100)

    plt.show()

def get_rotation_per_coverage(coverages, data):
    for alg_idx, algorithm in enumerate(algorithms):
        
        average_length = np.array([get_average_for_coverage(data, algorithm, "rotation", coverage) for coverage in coverages])        
        if not len(average_length):
            continue
        error = np.array([get_error_for_coverage(data, algorithm, "rotation", coverage) for coverage in coverages])   
        plt.plot(coverages, average_length, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(coverages, average_length-error, average_length+error, color=colors[alg_idx])

    plt.ylabel('Rotation [rad]')
    plt.xlabel('Coverage [%]')
    plt.legend()
    plt.title('Total Rotation per Coverage')
    #plt.xlim(0,300)
    #plt.ylim(0,100)

    plt.show()

def print_means(data, coverage):
    result = []
    for alg_idx, algorithm in enumerate(algorithms):
        average_length = get_average_for_coverage(data, algorithm, "length", coverage)
        error_length = get_error_for_coverage(data, algorithm, "length", coverage)
        average_rotation = get_average_for_coverage(data, algorithm, "rotation", coverage)
        error_rotation = get_error_for_coverage(data, algorithm, "rotation", coverage)  
        average_time = get_average_for_coverage(data, algorithm, "time", coverage)
        error_time = get_error_for_coverage(data, algorithm, "time", coverage)                      
        print("="*20)
        print(algorithm)
        print("Length: " + str(average_length) + "+- " + str(error_length))
        print("Rotation: " + str(average_rotation) + "+- " + str(error_rotation))
        print("Time: " + str(average_time) + "+- " + str(error_time))
        result.append({
            "Algorithm": algorithm,
            "average_length": average_length,
            "error_length": error_length,
            "average_rotation": average_rotation,
            "error_rotation": error_rotation,
            "average_time": average_time,
            "error_time": error_time
        })
    return result

def print_means_on_time(data, time):
    for alg_idx, algorithm in enumerate(algorithms):
        average_length = get_average(data, algorithm, "length", time)
        error_length = get_error(data, algorithm, "length", time)
        average_rotation = get_average(data, algorithm, "rotation", time)
        error_rotation = get_error(data, algorithm, "rotation", time)  
        average_time = get_average(data, algorithm, "coverage", time)
        error_time = get_error(data, algorithm, "coverage", time)                      
        print("="*20)
        print(algorithm)
        print("Length: " + str(average_length) + "+- " + str(error_length))
        print("Rotation: " + str(average_rotation) + "+- " + str(error_rotation))
        print("Time: " + str(average_time) + "+- " + str(error_time))

def get_length_randombastar(data, max_distances, other_alg_results):
    average = np.array([get_average_for_max_dist(data, "Random BAstar", "length", max_dist) for max_dist in max_distances])        
    error = np.array([get_error_for_max_dist(data, "Random BAstar", "length", max_dist) for max_dist in max_distances])   
    print(average)
    print(error)
    plt.plot(max_distances, average, 'm-.', label="Sampled BA* & Spiral")
    plt.fill_between(max_distances, average-error, average+error, color=(1.0,0.0,1.0,0.3))

    for alg_idx, algorithm_result in enumerate(other_alg_results):
        average = np.array([algorithm_result["average_length"]]*len(max_distances))
        error = np.array([algorithm_result["error_length"]]*len(max_distances))
        plt.plot(max_distances, average, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(max_distances, average-error, average+error, color=colors[alg_idx])

    plt.ylabel('Length [m]')
    plt.xlabel('$d_{max}$ [m]')
    plt.legend()
    plt.title('Length of path for different $d_{max}$')
    plt.show()

def get_rotation_randombastar(data, max_distances, other_alg_results):
    average = np.array([get_average_for_max_dist(data, "Random BAstar", "rotation", max_dist) for max_dist in max_distances])        
    error = np.array([get_error_for_max_dist(data, "Random BAstar", "rotation", max_dist) for max_dist in max_distances])   
    print(average)
    print(error)
    plt.plot(max_distances, average, 'm-.', label="Sampled BA* & Spiral")
    plt.fill_between(max_distances, average-error, average+error, color=(1.0,0.0,1.0,0.3))

    for alg_idx, algorithm_result in enumerate(other_alg_results):
        average = np.array([algorithm_result["average_rotation"]]*len(max_distances))
        error = np.array([algorithm_result["error_rotation"]]*len(max_distances))
        plt.plot(max_distances, average, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(max_distances, average-error, average+error, color=colors[alg_idx])

    plt.ylabel('Rotation [rad]')
    plt.xlabel('$d_{max}$ [m]')
    plt.legend()
    plt.title('Total Rotation for different $d_{max}$')
    plt.show()

def get_time_randombastar(data, max_distances, other_alg_results):
    average = np.array([get_average_for_max_dist(data, "Random BAstar", "time", max_dist) for max_dist in max_distances])        
    error = np.array([get_error_for_max_dist(data, "Random BAstar", "time", max_dist) for max_dist in max_distances])   
    print(average)
    print(error)
    plt.plot(max_distances, average, 'm-.', label="Sampled BA* & Spiral")
    plt.fill_between(max_distances, average-error, average+error, color=(1.0,0.0,1.0,0.3))

    for alg_idx, algorithm_result in enumerate(other_alg_results):
        average = np.array([algorithm_result["average_time"]]*len(max_distances))
        error = np.array([algorithm_result["error_time"]]*len(max_distances))
        plt.plot(max_distances, average, line[alg_idx], label=labels[alg_idx])
        plt.fill_between(max_distances, average-error, average+error, color=colors[alg_idx])

    plt.ylabel('Time [s]')
    plt.xlabel('$d_{max}$ [m]')
    plt.legend()
    plt.title('Computational time for different $d_{max}$')
    plt.show()


def main():
    with open('experiment_find_percentage.csv', newline='') as csvfile:
        data = csv.DictReader(csvfile)
        data = [row for row in data]
        times = np.array([int(row["time"]) for row in data])
        times = np.arange(0, max(times), 5)
        #coverages = np.arange(0, 101, 0.1)
        #other_alg_results = print_means(data, 98.3)
        print(data)
        #get_random_bastar_coverage(times, data)
        #get_length(times, data)
        #get_rotation(times, data)
        #print_means(data, 98.3)
        #print_means_on_time(data, 180)
        #print(get_average(data, "BAstar Variant", "length", 180))

    with open('experiment_official_2.csv', newline='') as csvfile:
        data = csv.DictReader(csvfile)
        data = [row for row in data]
        times = np.array([int(row["time"]) for row in data])
        times = np.arange(0, max(times), 10)
        coverages = np.arange(0, 101, 0.1)
        other_alg_results = print_means(data, 97.5)
        get_rotation_per_coverage(coverages, data)
        get_length_per_coverage(coverages, data)
        get_coverage(times, data)
        get_length(times, data)
        get_rotation(times, data)
        #print_means(data, 98.3)
        #print_means_on_time(data, 180)
        #print(get_average(data, "BAstar Variant", "length", 180))

    with open('randombastar_official.csv', newline='') as csvfile:
        data = csv.DictReader(csvfile)
        data = [row for row in data]
        max_distances = set( [float(row['Max dist']) for row in data] )
        max_distances = sorted(max_distances)
        print(max_distances)
        #get_length_randombastar(data, max_distances, other_alg_results)
        #get_rotation_randombastar(data, max_distances, other_alg_results)
        #get_time_randombastar(data, max_distances, other_alg_results)
        


        
        