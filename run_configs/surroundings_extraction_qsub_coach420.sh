#!/bin/bash
#PBS -N coach420
#PBS -l select=1:ncpus=32:mem=64gb:scratch_local=64gb
#PBS -l walltime=72:00:00
#PBS -j oe

# shellcheck disable=SC2164
cd "$SCRATCH"
git clone https://github.com/eleniel-mocna/Refined
# shellcheck disable=SC2164
cd Refined
module add python py-virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
mkdir data
cp -r /storage/brno12-cerit/home/eleniel/refined/data/extracted data
export PYTHONPATH=$PYTHONPATH:src
python3 src/dataset/surroundings_calculation/extract_surroundings.py coach420
cp -r data/surroundings /storage/brno12-cerit/home/eleniel/refined/coach420_result
