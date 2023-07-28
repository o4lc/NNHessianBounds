#!/bin/bash

echo 'Running Random Test'
python3 run.py RandomNetTanh
python3 run.py RandomNetSig

echo 'Running Linear Systems'
python3 run.py DoubleIntegratorS
python3 run.py quadrotorS

echo 'Running nonLinear Systems'
python3 run.py B1
python3 run.py B2
python3 run.py B3
python3 run.py B4
python3 run.py B5
python3 run.py TORA

