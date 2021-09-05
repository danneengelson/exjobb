

import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import copy

#def main():
with open('experiment_official_2_w_randombastar.csv', newline='') as csvfile:
    data = csv.DictReader(csvfile)
    data = [row for row in data]
    new_data = [data[0]]
    for idx, row in enumerate(data[1:]):
        print((idx, row))
        if row["algorithm"] in ["Inward Spiral", "BAstar", "BAstar Variant"]:
            new_data.append(row)
        elif float(row["coverage"]) > float(data[idx]["coverage"]) or row["time"] == "0":
            print("COVERAGE: " + str(row["coverage"]) + str("is bigger than") + str(data[idx]["coverage"]))
            new_data.append(row)
            latest = row
        else:
            time =  row["time"]
            copy_row = copy.deepcopy(latest)
            copy_row["time"] = time
            print("COPY ROW" + str(copy_row))
            new_data.append(copy_row)
        print("inserted: " + str(new_data[-1]))


    
with open('experiment_official_2_w_randombastar_fix.csv', 'w', newline='') as csvfile:
    fieldnames = ['point', 'algorithm', 'time', "coverage", "length", "rotation"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in new_data:
        writer.writerow(result)

