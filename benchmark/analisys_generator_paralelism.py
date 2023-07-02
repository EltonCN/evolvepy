import argparse
import sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_speedup, get_category_id

NvtxCategory = 33
NvtxStartEndRange = 60

names = ["Serial", "Parallel"]
base = "Serial"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("report_file", help="sqlite file created when running the benchmark")
    parser.add_argument("overhead_report_file", help="sqlite file created when running the benchmark with 'overhead' option")
    
    args = parser.parse_args()

    files = {"Paralelism":args.report_file, "Overhead":args.overhead_report_file}

    for experiment_name in files:
        con = sqlite3.connect(files[experiment_name])
        
        #Get category_id
        category_id = get_category_id(con, "benchmark")

        #Get times
        generator_times = {}
        generator_stds = {}
        for name in names:
            df_eval = pd.read_sql_query("SELECT * FROM NVTX_EVENTS WHERE text LIKE '{0}%' AND category = '{1}'".format(name, category_id), con)
            df_eval["population_size"] = df_eval.apply(lambda x: int(x["text"].split("_")[-1]), axis=1)
            df_eval.sort_values("population_size", inplace=True)

            sizes = np.unique(np.array(df_eval["population_size"]))
            times = []
            stds = []
            for size in sizes:
                mask = df_eval["population_size"] == size
                end = df_eval[mask]["end"].to_numpy()
                start = df_eval[mask]["start"].to_numpy()

                size_times = end-start
                times.append(np.mean(size_times))
                stds.append(np.std(size_times))

            generator_times[name] = np.array(times)
            generator_stds[name] = np.array(stds)


        #Plot times
        fig = plt.figure()
        for name in names:
            plt.errorbar(sizes, generator_times[name], generator_stds[name], fmt="o", capsize=5.0, markersize=5.0, label=name)

        plt.xticks(sizes)
        plt.legend()

        if experiment_name == "Overhead":
            plt.yscale("log")

        plt.suptitle("Generator Times")
        plt.title(experiment_name)
        fig.savefig("Generator_Paralelism_Times_{0}.png".format(experiment_name),  facecolor='white', transparent=False)

        fig = plt.figure()
        plot_speedup(sizes, generator_times, generator_stds, base)
        plt.suptitle("Generator Speedups")
        plt.title(experiment_name)
        plt.legend()
        if experiment_name == "Overhead":
            plt.yscale("log")
        fig.savefig("Generator_Paralelism_Speedups_{0}.png".format(experiment_name),  facecolor='white', transparent=False)