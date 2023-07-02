import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NvtxCategory = 33

def compute_u_speedup(base, u_base, accelerated, u_accelerated):
    u_speedup = np.power(u_accelerated, 2)*np.power(base, 2)/np.power(accelerated, 4)
    u_speedup += np.power(u_base, 2)/np.power(accelerated, 2)
    u_speedup = np.sqrt(u_speedup)

    return u_speedup

def plot_speedup(sizes, times, stds, base_name):
    for name in times:
        if name != base_name:
            speedup = times[base_name]/times[name]
            u_speedup = compute_u_speedup(times[base_name], stds[base_name], times[name], stds[name])
            plt.errorbar(sizes, speedup, u_speedup, fmt="o", capsize=5.0, markersize=5.0, label=name)
    
    plt.axhline(1, color="black")
    plt.xticks(sizes)

def get_category_id(con, category_name):
    df = pd.read_sql_query("SELECT * FROM NVTX_EVENTS", con)

    #Get category_id
    mask = df["eventType"] == NvtxCategory
    mask = np.bitwise_and(mask, df["text"] == category_name)
    row = df[mask]

    category_id = int(row["category"])

    return category_id
