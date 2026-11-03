#!/bin/bash
# Condor executable: run one chunk with the coffea conda env's python.
set -euo pipefail

PYTHON=/users/elarieke/miniconda3/envs/coffea/bin/python
HERE="/users/elarieke/stop_analysis/stp_cmp/DataMC/condor"

exec "$PYTHON" "$HERE/run_chunk.py" "$@"
