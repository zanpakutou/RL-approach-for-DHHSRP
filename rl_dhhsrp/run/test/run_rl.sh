#!/bin/bash
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 150 --instance_type uniform --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 240 --instance_type uniform --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 360 --instance_type uniform --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 150 --instance_type uniform --nb_nurse 6 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 240 --instance_type uniform --nb_nurse 6 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 360 --instance_type uniform --nb_nurse 6 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 150 --instance_type cluster --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 240 --instance_type cluster --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 360 --instance_type cluster --nb_nurse 1 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 150 --instance_type cluster --nb_nurse 6 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 240 --instance_type cluster --nb_nurse 6 --output_folder result_rl
python3 evaluate.py --obj patient --nb_scenario 1 --arr_rate 360 --instance_type cluster --nb_nurse 6 --output_folder result_rl
