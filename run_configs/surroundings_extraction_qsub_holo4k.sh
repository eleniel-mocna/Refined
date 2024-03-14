#!/bin/bash
#PBS -N holo4k
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=64gb
#PBS -l walltime=48:00:00
#PBS -j oe

cd $SCRATCH
cp /storage/brno12-cerit/home/eleniel/refined .
cd refined
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 python3 src/dataset/surroundings_calculation/extract_surroundings.py holo4k
cp -r data /storage/brno12-cerit/home/eleniel/refined/holo4k_result
