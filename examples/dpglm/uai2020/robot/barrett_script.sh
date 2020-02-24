#!/bin/sh

python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 500 --gibbs_iters 100 --stochastic --seed 1 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 500 --gibbs_iters 100 --stochastic --seed 2 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 500 --gibbs_iters 100 --stochastic --seed 3 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 500 --gibbs_iters 100 --stochastic --seed 4 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 500 --gibbs_iters 100 --stochastic --seed 5 --nb_models 1000 --mute

python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 1000 --gibbs_iters 100 --stochastic --seed 1 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 1000 --gibbs_iters 100 --stochastic --seed 2 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 1000 --gibbs_iters 100 --stochastic --seed 3 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 1000 --gibbs_iters 100 --stochastic --seed 4 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior stick-breaking --alpha 1000 --gibbs_iters 100 --stochastic --seed 5 --nb_models 1000 --mute

python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 100 --stochastic --seed 1 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 100 --stochastic --seed 2 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 100 --stochastic --seed 3 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 100 --stochastic --seed 4 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 100 --stochastic --seed 5 --nb_models 1000 --mute

python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 10 --gibbs_iters 100 --stochastic --seed 1 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 10 --gibbs_iters 100 --stochastic --seed 2 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 10 --gibbs_iters 100 --stochastic --seed 3 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 10 --gibbs_iters 100 --stochastic --seed 4 --nb_models 1000 --mute &
python evaluate_barrett.py --nb_train 10000 --prior dirichlet --alpha 10 --gibbs_iters 100 --stochastic --seed 5 --nb_models 1000 --mute
