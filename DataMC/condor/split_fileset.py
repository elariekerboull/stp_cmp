"""
Build the full MC+Data fileset (same logic as run_MC_Data.py) and split it
into N roughly-equal chunks by file count, one chunk per condor job.

Each chunk is written as condor/chunks/chunk_<i>.json, a dict of
{dataset_name: [files...]} covering a slice of every dataset so that no
single job is stuck with one huge dataset while others sit idle.

Usage:
    python split_fileset.py --njobs 6
"""
import argparse
import json
import os

SAMPLES_PATH = "/users/scormenier/stp/Datasets/samples.json"
CHOSEN_YEAR = "2018"
HERE = os.path.dirname(os.path.abspath(__file__))

def build_fileset(samples, chosen_year=CHOSEN_YEAR):
    fileset = {}

    for process in samples["MC"]:
        for sample in samples["MC"][process]:
            for year, info in samples["MC"][process][sample].items():
                if year == chosen_year:
                    dataset_name = f"{process}__{sample}__{year}"
                    fileset.setdefault(dataset_name, []).extend(info["files"])

    for year in samples["Data"]:
        if year == chosen_year:
            for run_letter, info in samples["Data"][year].items():
                dataset_name = f"data__{run_letter}__{year}"
                fileset.setdefault(dataset_name, []).extend(info["files"])

    return fileset


def split_fileset(fileset, njobs):
    # Flatten to (dataset, file) pairs, then round-robin across njobs chunks
    # so every job gets a slice of every (large) dataset instead of whole
    # datasets being assigned unevenly.
    pairs = [(ds, f) for ds, files in fileset.items() for f in files]

    chunks = [dict() for _ in range(njobs)]
    for i, (ds, f) in enumerate(pairs):
        chunk = chunks[i % njobs]
        chunk.setdefault(ds, []).append(f)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument("--year", default=CHOSEN_YEAR)
    args = parser.parse_args()

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    fileset = build_fileset(samples, args.year)
    total_files = sum(len(v) for v in fileset.values())
    print(f"Total datasets: {len(fileset)}, total files: {total_files}")

    chunks = split_fileset(fileset, args.njobs)

    chunk_dir = os.path.join(HERE, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        n = sum(len(v) for v in chunk.values())
        path = os.path.join(chunk_dir, f"chunk_{i}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        print(f"  chunk_{i}.json: {n} files across {len(chunk)} datasets")


if __name__ == "__main__":
    main()
