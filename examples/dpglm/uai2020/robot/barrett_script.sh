#!/bin/sh

python evaluate_barrett.py --nb_train 5000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 512 --seed 1 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 512 --seed 2 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 512 --seed 3 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 512 --seed 4 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 512 --seed 5 --nb_models 500 --mute &

python evaluate_barrett.py --nb_train 5000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 512 --seed 1 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 512 --seed 2 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 512 --seed 3 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 512 --seed 4 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 5000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 512 --seed 5 --nb_models 500 --mute

python evaluate_barrett.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 1 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 2 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 3 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 4 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 5 --nb_models 500 --mute &

python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 1 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 2 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 3 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 4 --nb_models 500 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 5 --nb_models 500 --mute
