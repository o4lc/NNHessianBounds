#!/bin/bash

# Input 1: Lip Method -> 0:Naive, 1:LipSDP, 2:LipLT

 
# ## Fig. 6
# echo 'Running Random Test'
# python3 run.py --config RandomNetTanh --eps 1e-2 --lipMethod $1
# python3 run.py --config RandomNetSig --eps 1e-2 --lipMethod $1

# ## Fig. 8
# echo 'Running Linear Systems'
# python3 run.py --config DoubleIntegratorS --eps 1e-2 --lipMethod 2
# echo '------------------'
# python3 run.py --config quadrotorS --eps 1e-2 --lipMethod 2


# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 0
# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 1
# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 2
# echo '------------------'
# python3 run.py --config quadrotorS --eps 1e-2 --lipMethod 0
# python3 run.py --config quadrotorS --eps 1e-2 --lipMethod 1
# python3 run.py --config quadrotorS --eps 1e-2 --lipMethod 2

# python3 run.py --config quadrotorS --eps 1e-1 --lipMethod 0
# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 1 --splittingMethod 'BestLB'
# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 1 --splittingMethod 'length'
# python3 run.py --config quadrotorS --eps 1e-3 --lipMethod 2


python3 run.py --config DoubleIntegratorS --eps 1e-2 --lipMethod 2 --splittingMethod 'BestLB'
python3 run.py --config DoubleIntegratorS --eps 1e-2 --lipMethod 2 --splittingMethod 'length'
# echo '------------------'
python3 run.py --config DoubleIntegratorS --eps 1e-3 --lipMethod 2 --splittingMethod 'BestLB'
python3 run.py --config DoubleIntegratorS --eps 1e-3 --lipMethod 2 --splittingMethod 'length'