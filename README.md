# HiSS

This repository contains code for the paper Slithering through Gaps: Capturing Discrete Isolated Modes via
Logistic Bridging, accepted in International Conference on Artificial Intelligence and Statistics (AISTATS), 2026.


```bibtex
@article{mohanty2026hiss,
  title={Slithering through Gaps: Capturing Discrete Isolated Modes via
Logistic Bridging},
  author={Mohanty, Pinaki and Zhang, Ruqi},
  journal={International Conference on Artificial Intelligence and Statistics},
  year={2026}
}
```


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
python TSP.py --sampler=<SAMPLER> 
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
