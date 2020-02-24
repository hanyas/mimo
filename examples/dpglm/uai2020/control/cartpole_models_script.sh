#!/bin/sh

python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 50 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 75 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 100 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 125 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 150 --mute &

python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 50 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 75 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 100 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 125 --mute &
python evaluate_cartpole.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 150 --mute
