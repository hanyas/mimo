#!/bin/sh

python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 1 --mute &
python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 5 --mute &
python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 10 --mute &
python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 15 --mute &
python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 20 --mute &
python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 30 --horizon 25 --mute

#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute &
#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute &
#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute &
#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute &
#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute &
#python evaluate_pendulum.py --name horizon --gibbs_iters 100 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 30 --mute
