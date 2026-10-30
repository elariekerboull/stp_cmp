"""
Condor worker script: process one fileset chunk (produced by split_fileset.py)
and save the resulting histograms to its own .coffea file. Run once per
condor job; outputs are merged afterwards by merge_outputs.py.

Output path is derived from the chunk filename: chunks/chunk_3.json -> out/chunk_3.coffea

Usage:
    run_chunk.py <chunk.json> [nworkers]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from analysis_MC_Data import StopAnalysisMC_Data
import coffea.util as util

SAMPLES_PATH = "/users/scormenier/stp/Datasets/samples.json"

def kill_child_processes():
    """Kill any leftover child processes spawned by this script."""
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        child.kill()  # force kill stragglers

def main():
    chunk_path = sys.argv[1]
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("OMP_NUM_THREADS", 4))

    stem = os.path.splitext(os.path.basename(chunk_path))[0]
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{stem}.coffea")

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    with open(chunk_path) as f:
        fileset = json.load(f)

    processor_instance = StopAnalysisMC_Data(samples)

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=nworkers),
        schema=NanoAODSchema,
        skipbadfiles=True,
    )


    try:
        output = runner(
            fileset,
            treename="Events",
            processor_instance=processor_instance,
        )
        util.save(output, output_path)
        print("Output saved to {output_path}")
        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted — cleaning up worker processes...")
        kill_child_processes()
        sys.exit(1)

    finally:
        kill_child_processes()


if __name__ == "__main__":
    main()
