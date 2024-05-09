#!/bin/bash
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install --no-cache-dir --upgrade pip setuptools
pip3 install --no-cache-dir -r requirements.txt
export export PYTHONPATH=$PYTHONPATH:src
python3 src/models/run_scripts/whole_pipeline.py
