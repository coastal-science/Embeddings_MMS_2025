#!/usr/bin/env bash
# Runs the full pipeline (preprocessor -> model_trainer -> embeddings) once for
# the SimCLR config and once for the supervised classifier config.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== SimCLR pipeline ==="
run-pipeline --config configs/simclr.yaml

echo "=== Supervised classifier pipeline ==="
run-pipeline --config configs/supervised.yaml
