#!/bin/bash
#SBATCH --job-name=duke_aml_preextract_cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="/mnt/bulk-neptune/nguyenmin/stamp-dev/Slide-Agent/temp/Pathology_agent"
EXPERIMENT_ROOT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/aml_gemma4_embedding_suite"
TILE_FILTER="hybrid"
EXTRACTORS="uni2"

cd "${REPO_ROOT}"
source ~/venv/bin/activate
echo "Starting AML preextract cache suite"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
date

bash evaluate/preextract_hybrid_cache_suite.sh \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --tile-filter "${TILE_FILTER}" \
  --extractors "${EXTRACTORS}"

date
echo "AML preextract cache suite finished"
