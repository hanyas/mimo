#!/bin/sh

python evaluate_alpha.py --dataset cmb --nb_models 100
python evaluate_alpha.py --dataset sine --nb_models 100
python evaluate_alpha.py --dataset fk_1joint --nb_models 100
python evaluate_alpha.py --dataset fk_3joint --nb_models 100
python evaluate_alpha.py --dataset sarcos --nb_models 100

python evaluate_alpha.py --dataset cmb --nb_models 100 --prior stick-breaking
python evaluate_alpha.py --dataset sine --nb_models 100 --prior stick-breaking
python evaluate_alpha.py --dataset fk_1joint --nb_models 100 --prior stick-breaking
python evaluate_alpha.py --dataset fk_3joint --nb_models 100 --prior stick-breaking
python evaluate_alpha.py --dataset sarcos --nb_models 100 --prior stick-breaking

