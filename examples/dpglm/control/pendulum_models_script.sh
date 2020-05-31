#!/bin/sh

python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 50 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 150 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 200 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 250 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 300 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 350 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 400 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 450 --mute &
python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 500 --mute

#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 50 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 150 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 200 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 250 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 300 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 350 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 400 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 450 --mute &
#python evaluate_pendulum.py --task models --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 500 --mute
