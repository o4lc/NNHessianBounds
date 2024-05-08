#!/bin/bash

# Input 1: Lip Method -> 0:Naive, 1:LipSDP, 2:LipLT

 
echo 'Running Random Test'
## Fig. 6
python3 run.py --config RandomNetTanh --eps 1e-2 --lipMethod 2 --iterations 1
python3 run.py --config RandomNetSig --eps 1e-2 --lipMethod 2 --iterations 1
## Fig. 7
python3 run.py --config RandomNetTanh1 --eps 1e-2 --lipMethod 2 --iterations 1
python3 run.py --config RandomNetTanh2 --eps 1e-2 --lipMethod 2 --iterations 1
python3 run.py --config RandomNetTanh3 --eps 1e-2 --lipMethod 2 --iterations 1


echo 'Running Linear Systems'
## Fig. 7
python3 run.py --config DoubleIntegratorS --eps 1e-2 --lipMethod 2 --iterations 1
python3 run.py --config quadrotorS --eps 1e-2 --lipMethod 2 --iterations 1



