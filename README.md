# HiSS




# Introduction
We propose Hiss, a discrete sampler for sampling from landscapes with disconnected modes.


# Dependencies
* [PyTorch 1.9.1](http://pytorch.org/) 
* [torchvision 0.10.1](https://github.com/pytorch/vision/)

# Usage

## Sampling From 4D Joint Bernoulli
Enter Directory
```
./Bernoulli
```
Then run
```
python bernoulli_sample.py --sampler=<SAMPLER>
```

## Sampling From Ising Models
Please run
```
python ising_sample.py --sampler=<SAMPLER>
```

## Travelling Salesman Problem
Enter Directory
```
./TSP
```
Then run
```
python TSPpw.py --sampler=<SAMPLER> 
```

## Binary Bayesian Neural Networks
Enter Directory
```
./BinaryBNN
```
Then run
```
python bayesian_nn.py --sampler=<SAMPLER> --dataset=<DATASET>
```

# References
* This repo is built upon the [DLP repo](https://github.com/ruqizhang/discrete-langevin) 
