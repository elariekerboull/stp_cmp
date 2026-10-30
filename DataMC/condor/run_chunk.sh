#!/bin/bash
# Condor executable: run one chunk with the coffea conda env's python.
set -euo pipefail

PYTHON=/users/scormenier/miniconda3/envs/coffea/bin/python
HERE="/users/scormenier/stp/MC_Data/condor"

exec "$PYTHON" "$HERE/run_chunk.py" "$@"
