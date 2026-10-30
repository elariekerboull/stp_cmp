import coffea.util as util
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import json

year = "2018"
input_file = "output_MC_Data-merged.coffea"

# --------------------------------------------------------
# NEW: choose what to plot here
# --------------------------------------------------------
plot_mode = "data"   # "MC" or "data"

output_dir = f"plots_{year}/2Dplots"
os.makedirs(output_dir, exist_ok=True)
output = util.load(input_file)

with open("../Datasets/samples.json") as f:
    samples = json.load(f)

lumi = samples["luminosity"][year] * 1000  # /pb


def get_processes_for_mode(histogram, mode):
    """
    Split the 'process' axis into MC processes vs the data process(es),
    same convention as the data/MC agreement script: data datasets are
    always named 'data__<run_letter>__<year>', so process == 'data'.
    """
    all_procs = [str(p) for p in histogram.axes["process"]]
    if mode == "data":
        procs = [p for p in all_procs if p.startswith("data")]
    elif mode == "MC":
        procs = [p for p in all_procs if not p.startswith("data")]
    else:
        raise ValueError(f"Unknown plot_mode '{mode}', expected 'MC' or 'data'")

    if not procs:
        print(f"WARNING: no processes found for mode='{mode}' "
              f"(available: {all_procs})")
    return procs


def plot_histogram_2d(histogram, variable_name, processes_to_sum, mode,
                       vmin=6e-4, vmax=1.5e2):
    """
    Sum only `processes_to_sum` (already filtered to MC-only or data-only
    upstream) and plot the resulting 2D histogram.
    """
    h_year = histogram[{"year": year}]

    h_total = None
    for process in processes_to_sum:
        h_proc = h_year[{"process": process}]
        h_total = h_proc if h_total is None else h_total + h_proc

    if h_total is None:
        print(f"WARNING: No processes found for {variable_name} (mode={mode}), skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    h_total.plot2d(ax=ax)#, norm=LogNorm(vmax = 4.7e2))

    label = "Data" if mode == "data" else "All MC processes summed"
    ax.set_title(f"{variable_name} - {label} ({year})")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{variable_name}_{year}_{mode}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


variables_to_plot = [
    "2D_jet1_eta_phi_postHEM_110_200", "2D_jet1_eta_phi_postHEM_200_300", "2D_jet1_eta_phi_postHEM_300_400", "2D_jet1_eta_phi_postHEM_400_500", 
]

signals = ["4BD-500-490", "4BD-500-420"]

for var in variables_to_plot:
    processes_to_sum = get_processes_for_mode(output[var], plot_mode)
    plot_histogram_2d(output[var], var, processes_to_sum, plot_mode)