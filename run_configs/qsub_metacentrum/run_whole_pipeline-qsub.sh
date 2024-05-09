#!/bin/bash
#PBS -N REFINED_CNN_GPU
#PBS -q gpu -l select=1:ncpus=2:ngpus=1:mem=64gb:scratch_local=256gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS

# shellcheck disable=SC2164
cd "$SCRATCH"
git clone https://github.com/eleniel-mocna/Refined
# shellcheck disable=SC2164
cd Refined
module add python/python-3.10.4-intel-19.0.4-sc7snnf
python3 -m venv CHOSEN_VENV_DIR
source CHOSEN_VENV_DIR/bin/activate
CHOSEN_VENV_DIR/bin/pip install --no-cache-dir --upgrade pip setuptools
pip3 install --no-cache-dir -r requirements.txt

export PYTHONPATH=$PYTHONPATH:src
python3 src/models/run_scripts/whole_pipeline.py