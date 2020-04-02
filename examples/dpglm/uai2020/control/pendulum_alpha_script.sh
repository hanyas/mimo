#!/bin/sh

python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 1.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 10.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 100.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 500.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior stick-breaking --alpha 1000.0 --nb_models 100 --mute

python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 0.1 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 1.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 50.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 100.0 --nb_models 100 --mute &
python evaluate_pendulum.py --task alpha --gibbs_iters 1 --stochastic --nb_seeds 25 --prior dirichlet --alpha 500.0 --nb_models 100 --mute
