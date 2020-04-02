#!/bin/sh

#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 1 --mute &
#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 5 --mute &
#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 10 --mute &
#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 15 --mute &
#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 20 --mute &
#python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --horizon 25 --mute

python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 1 --mute &
python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 5 --mute &
python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 10 --mute &
python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 15 --mute &
python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 20 --mute &
python evaluate_cartpole.py --task horizon --gibbs_iters 100 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --horizon 25 --mute
