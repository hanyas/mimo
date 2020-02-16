#!/bin/sh

#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 1 --mute &
#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 5 --mute &
#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 10 --mute &
#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 15 --mute &
#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 20 --mute &
#python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior stick-breaking --alpha 10.0 --nb_models 100 --horizon 25 --mute

python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 1 --mute &
python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 5 --mute &
python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 10 --mute &
python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 15 --mute &
python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 20 --mute &
python evaluate_cartpole.py --name horizon --gibbs_iters 1 --nb_seeds 10 --prior dirichlet --alpha 1.0 --nb_models 50 --horizon 25 --mute
