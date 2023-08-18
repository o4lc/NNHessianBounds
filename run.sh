#!/bin/bash

echo 'Running Random Test'
python3 run.py --config RandomNetTanh
python3 run.py --config RandomNetSig

echo 'Running Linear Systems'
python3 run.py --config DoubleIntegratorS
python3 run.py --config quadrotorS

echo 'Running nonLinear Systems'
python3 run.py --config B1
python3 run.py --config B2
python3 run.py --config B3
python3 run.py --config B4
python3 run.py --config B5
python3 run.py --config TORA

