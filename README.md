# penetration_test
Official repository of penetration test by reinforcement learning. This repository is built on the open-source penetration testing benchmark: [NetworkAttackSimulator](https://github.com/Jjschwartz/NetworkAttackSimulator).


## Installation

* wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
* bash Anaconda3-2024.10-1-Linux-x86_64.sh
* export PATH = $PATH:~/anaconda3/bin
* conda env create -f environment.yml
* conda activate nasim

## Training

For large-scale training, please refer to the follwing scripts:
* sh train.sh

For single training on different scenarios, please refer to follwing scripts:
* python train_constrained_dqn.py tiny policies/CDQN.pth

## Test

For testing the trained policies on different scenarios, please refer to follwing scripts:
* python run_dqn_policy.py tiny policies/CDQN.pth -seed 0


# Hyperparameters for DQN and Constrained-DQN

| **Hyperparameters** | **Settings**      |
|---------------------|-------------------|
| Network             | [256,512,256]     |
| Activation          | ReLU              |
| Learning Rate       | 1 × 10⁻³          |
| Replay Buffer       | 10,000            |
| Batch Size          | 128               |
| Optimizer           | Adam              |


## Associated Publication

This repository supports the findings of the following paper:

> **[Multi-Objective Policy Optimization for Effective and Cost-Conscious Penetration Testing]**  
> Authors: [Xiaojuan Cai], [Lulu Zhu], [Zhuo Li], [Hiroshi Koide]  
> Citation: Cai, X.; Zhu, L.; Li, Z. and Koide, H. (2025). Multi-Objective Policy Optimization for Effective and Cost-Conscious Penetration Testing.  In Proceedings of the 21st International Conference on Web Information Systems and Technologies, ISBN 978-989-758-772-6, ISSN 2184-3252, pages 488-499.    

For additional implementation details, please refer to:

`Appendices_Implement_Details/` – contains environment components, a simplified example of constrained policy optimization, and implementation notes of Random- and Rule-Based Method not included in the main paper.


