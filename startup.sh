#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
nohup python experiments_fedavg.py --model=simple-cnn --dataset=cifar10 --alg=fedavg --lr=0.01 --batch-size=64 --epochs=10 --n_parties=10 --rho=0.9 --comm_round=50 --partition=noniid-labeldir --beta=0.5 --device=cuda:0 --datadir=./data/ --logdir=./logs/ --noise=0.0 --init_seed=0 > experiments_fedavg.log 2>&1& echo $! > ./run.pid
echo start train_tsp
