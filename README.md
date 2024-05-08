# NNHessianBounds

This code is for the Experiments of "Provable Bounds on the Hessian of Neural Networks". 
And it is based on [ReachLipBnB](https://github.com/o4lc/ReachLipBnB).


# Installation Requirements

```
pip install torch
pip install cvxpy
pip install polytope
```

# Usage

Use `run.sh` for all functions. Refer to `run.py` for the usage of all the arguments.   
```
bash run.sh 
```

# Reproduce

To reproduce the experiments, following template can be used.

``` bash
python3 run.py --config DoubleIntegratorS --eps 1e-2 --lipMethod 2 --iterations 5
```
