#!/bin/sh

python evaluate_sarcos.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 1024 --seed 5 --nb_models 500 --mute &

python evaluate_sarcos.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 10000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 1024 --seed 5 --nb_models 500 --mute

python evaluate_sarcos.py --nb_train 20000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 2048 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 2048 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 2048 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 2048 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 2048 --seed 5 --nb_models 500 --mute &

python evaluate_sarcos.py --nb_train 20000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 2048 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 2048 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 2048 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 2048 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 20000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 2048 --seed 5 --nb_models 500 --mute

python evaluate_sarcos.py --nb_train 30000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 3072 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 3072 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 3072 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 3072 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 3072 --seed 5 --nb_models 500 --mute &

python evaluate_sarcos.py --nb_train 30000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 3072 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 3072 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 3072 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 3072 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 30000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 3072 --seed 5 --nb_models 500 --mute

python evaluate_sarcos.py --nb_train 40000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 4096 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 4096 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 4096 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 4096 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior stick-brekaing --alpha 100 --gibbs_iters 500 --svi_batchsize 4096 --seed 5 --nb_models 500 --mute &

python evaluate_sarcos.py --nb_train 40000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 4096 --seed 1 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 4096 --seed 2 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 4096 --seed 3 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 4096 --seed 4 --nb_models 500 --mute &
python evaluate_sarcos.py --nb_train 40000 --prior dirichlet --alpha 1 --gibbs_iters 500 --svi_batchsize 4096 --seed 5 --nb_models 500 --mute
