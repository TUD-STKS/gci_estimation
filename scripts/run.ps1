# "Glottal Closure Instant Detection using Echo State Networks"
#
# Copyright (C) 2022 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv .virtualenv

.\.virtualenv\Scripts\activate.ps1
python.exe -m pip install -r requirements.txt
python.exe .\src\main.py --plot --serialize

deactivate
