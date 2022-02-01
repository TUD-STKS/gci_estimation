#!/bin/sh

# "Glottal Closure Instant Detection using Echo State Networks"
#
# Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python3 -m venv .virtualenv
source .virtualenv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/main.py --plot --serialize

deactivate
