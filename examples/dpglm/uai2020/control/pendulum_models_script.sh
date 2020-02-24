#!/bin/sh

python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 30 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 45 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 60 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 75 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior stick-breaking --alpha 50.0 --nb_models 90 --mute &

python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 30 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 45 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 60 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 75 --mute &
python evaluate_pendulum.py --name models --gibbs_iters 100 --nb_seeds 25 --prior dirichlet --alpha 10.0 --nb_models 90 --mute
