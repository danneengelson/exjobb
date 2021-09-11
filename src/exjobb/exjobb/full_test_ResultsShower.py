import pprint
import numpy as np
import matplotlib.pyplot as plt

class ResultShower:
    def __init__(self, data):
        self.data = data

    def get_average_and_error(self, data, property, sample, sample_property):
        values = []
        nbr_of_points = len(data)
        for point_data in data:
            closest_to_sample = min(point_data, key = lambda i: abs(i[sample_property] - sample))

            if property == "cost":
                values.append(closest_to_sample["length"] + closest_to_sample["rotation"])
            else:
                values.append(closest_to_sample[property])

        return np.mean(np.array(values)), 2*np.std(np.array(values))/np.sqrt(nbr_of_points)

    def get_hyper_parameter_data(self, parameter):
        data_per_algorithm = []
        for algorithm in self.data.values():
            hyper_results_for_alg = algorithm["hyper_data"]
            if not hyper_results_for_alg:
                continue

            data_for_alg = []
            for evaluation in hyper_results_for_alg:
                if evaluation["result"]["status"] == "fail":
                    continue
                if parameter not in evaluation["misc"]["vals"]:
                    continue
                data_for_alg.append({
                    "cost": evaluation["result"]["loss"],
                    "value": evaluation["misc"]["vals"][parameter][0]
                })
            if data_for_alg:
                data_per_algorithm.append({
                        "data": data_for_alg,
                        "algorithm": algorithm["name"]
                    })
        return data_per_algorithm


    def plot_data(self, samples, property, sample_property):
        for algorithm in self.data.values():
            all_results_for_alg = algorithm["experiment_results"]
            if not all_results_for_alg:
                continue
            
            results = np.array([self.get_average_and_error(all_results_for_alg, property, sample, sample_property) for sample in samples])
            average, confidence_error = results[:,0], results[:,1]
            plt.plot(samples, average, algorithm["line"], label=algorithm["name"])
            plt.fill_between(samples, average-confidence_error, average+confidence_error, color=algorithm["confidence_color"])

    def show_coverage_per_time(self, max_time, step_size):
        time_samples = np.arange(0, max_time, step_size)
        self.plot_data(time_samples, "coverage", "time")
        plt.ylabel('Coverage [%]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.title('Coverage over Time')
        plt.xlim(0,max_time)
        plt.show()   

    def show_length_per_time(self, max_time, step_size):
        time_samples = np.arange(0, max_time, step_size)
        self.plot_data(time_samples, "length", "time")
        plt.ylabel('Length [m]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.title('Length over Time')
        plt.xlim(0,max_time)
        plt.show()

    def show_rotation_per_time(self, max_time, step_size):
        time_samples = np.arange(0, max_time, step_size)
        self.plot_data(time_samples, "rotation", "time")
        plt.ylabel('Rotation [rad]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.title('Rotation over Time')
        plt.xlim(0,max_time)
        plt.show()   
    
    def show_length_per_coverage(self, step_size):
        coverage_samples = np.arange(0, 101, step_size)
        self.plot_data(coverage_samples, "length", "coverage")
        plt.ylabel('Length [m]')
        plt.xlabel('Coverage [%]')
        plt.legend()
        plt.title('Length per Coverage')
        plt.xlim(0,100)
        plt.show()   

    def show_rotation_per_coverage(self, step_size):
        coverage_samples = np.arange(0, 101, step_size)
        self.plot_data(coverage_samples, "rotation", "coverage")
        plt.ylabel('Rotation [rad]')
        plt.xlabel('Coverage [%]')
        plt.legend()
        plt.title('Rotation per Coverage')
        plt.xlim(0,100)
        plt.show()   

    def show_cost_per_coverage(self, step_size):
        coverage_samples = np.arange(0, 101, step_size)
        self.plot_data(coverage_samples, "cost", "coverage")
        plt.ylabel('Cost')
        plt.xlabel('Coverage [%]')
        plt.legend()
        plt.title('Cost (Length + Rotation) per Coverage')
        plt.xlim(0,100)
        plt.show()   
    
    def show_cost_per_time(self, max_time, step_size):
        time_samples = np.arange(0, max_time, step_size)
        self.plot_data(time_samples, "cost", "time")
        plt.ylabel('Cost')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.title('Cost (Length + Rotation) over Time')
        plt.xlim(0, max_time)
        plt.show()   

    def show_hyper_parameter(self, parameter):
        data = self.get_hyper_parameter_data(parameter)

        formatted_data = []
        xticklabels = []
        for algorithm_data in data:
            xticklabels.append(algorithm_data["algorithm"])
            data_values = algorithm_data["data"]
            values = np.array( [ evalutaion["value"] for evalutaion in data_values])
            costs = np.array( [ evalutaion["cost"] for evalutaion in data_values] )
            #formatted_data.append(np.vstack((values, costs)).T)
            formatted_data.append(values)
            print(algorithm_data["algorithm"] + str(len(values)))

        fig, ax = plt.subplots()
        print(formatted_data)
        if len(formatted_data) == 1:
            ax.violinplot(formatted_data[0])
        else:
            ax.violinplot(formatted_data)

        ax.set_xticks(np.arange(1, len(xticklabels)+1))
        ax.set_xticklabels(xticklabels)
        
        plt.ylabel(parameter)
        plt.xlabel('Algorithm')
        plt.legend()
        plt.title("Hyper optimization of " + parameter)
        plt.show()   