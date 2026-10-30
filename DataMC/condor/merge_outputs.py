"""
Merge all per-chunk .coffea outputs (out/chunk_*.coffea) produced by the
condor jobs into a single output file, in the same format run_MC_Data.py
produces, so plot_MC_Data.py can consume it unchanged.

Usage:
    python merge_outputs.py [output_name]
"""
import glob
import os
import sys

import coffea.util as util
from coffea.processor import accumulate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "output_MC_Data-merged.coffea"

    chunk_outputs = sorted(glob.glob(os.path.join(HERE, "out", "chunk_*.coffea")))
    if not chunk_outputs:
        sys.exit("No chunk outputs found in out/ — did the condor jobs finish?")

    print(f"Merging {len(chunk_outputs)} chunk outputs:")
    for p in chunk_outputs:
        print(f"  {p}")

    results = [util.load(p) for p in chunk_outputs]
    merged = accumulate(results)

    final_path = os.path.join(os.path.dirname(HERE), out_name)
    util.save(merged, final_path)
    print(f"Merged output saved to {final_path}")


if __name__ == "__main__":
    main()
