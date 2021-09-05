import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt

def get_average(data, d_max, property, time):

    rows = list(filter(lambda row: int(row['max_distance']) == d_max and int(row['time']) == time, data))
    
    values = [float(row[property]) for row in rows]
    return np.mean(values)

def get_error(data, algorithm, property, time):
    #print(data)
    #rows2 = list(filter(lambda row:  int(row["time"]) == time, data))

    rows = list(filter(lambda row: int(row['max_distance']) == algorithm and int(row['time']) == time, data))
    
    #print(rows)
    values = [float(row[property]) for row in rows]
    return np.std(values)

def get_random_bastar_coverage(times, data):
    for d_max_idx, d_max in enumerate(max_dist):
        average_coverage = 100*np.array([get_average(data, d_max, "coverage", time) for time in times])        
        error = 100*np.array([get_error(data, d_max, "coverage", time) for time in times])   
        plt.plot(times, average_coverage, line[d_max_idx], label=labels[d_max_idx])
        #plt.fill_between(times, average_coverage-error, average_coverage+error, color=colors[d_max_idx])

    plt.ylabel('Coverage [%]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Coverage over Time')
    plt.xlim(0,300)
    plt.ylim(0,105)

    plt.show()

def get_random_bastar_length(times, data):
    for d_max_idx, d_max in enumerate(max_dist):
        
        average_length = np.array([get_average(data, d_max, "length", time) for time in times])        
        if not len(average_length):
            continue
        error = np.array([get_error(data, d_max, "length", time) for time in times])   
        plt.plot(times, average_length, line[d_max_idx], label=labels[d_max_idx])
        plt.fill_between(times, average_length-error, average_length+error, color=colors[d_max_idx])

    plt.ylabel('Length [m]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Length of Path over Time')
    plt.xlim(0,300)
    #plt.ylim(0,100)

    plt.show()

def get_random_bastar_rotation(times, data):
    for d_max_idx, d_max in enumerate(max_dist):
        
        average_rotation = np.array([get_average(data, d_max, "rotation", time) for time in times])        
        if not len(average_rotation):
            continue
        error = np.array([get_error(data, d_max, "rotation", time) for time in times])   
        plt.plot(times, average_rotation, line[d_max_idx], label=labels[d_max_idx])
        plt.fill_between(times, average_rotation-error, average_rotation+error, color=colors[d_max_idx])

    plt.ylabel('Rotation [rad]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Total Rotation over Time')
    plt.xlim(0,300)
    #plt.ylim(0,100)

    plt.show()

max_dist = [1,2,3,4,5,6,7]
line = ["*c-", "b", "r:", "g--", "m-.", "^k-", "oy-"]
labels = ["1", "2","3","4","5","6","7"]
colors = [(0.0,1.0,1.0,0.3), (0.0,0.0,1.0,0.3), (1.0,0.0,0.0,0.3), (0,1.0,0,0.3), (1.0,0,1.0,0.3), (0,0,0,0.3), (1.0,1.0,0,0.3)]  


def main():
    with open('experiment_find_percentage_1.csv', newline='') as csvfile:
        data = csv.DictReader(csvfile)
        data = [row for row in data]
        print(data)
        times = np.array([int(row["time"]) for row in data])
        times = np.arange(0, max(times), 5)
        get_random_bastar_coverage(times, data)
        get_random_bastar_length(times, data)
        get_random_bastar_rotation(times, data)