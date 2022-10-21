#!/bin/bash
#SBATCH --array=1
#SBATCH --job-name=10_11_16
#SBATCH --time=24:15:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --nodelist=mpu5
#SBATCH --mail-user=phamtusan@gmail.com
sh parallel_commands.sh "sh run_ddqn_1.sh 0" "sh run_ddqn_1.sh 1" "sh run_ddqn_1.sh 2" "sh run_ddqn_1.sh 3" "sh run_ddqn_2.sh 0" "sh run_ddqn_2.sh 1" "sh run_ddqn_2.sh 2" "sh run_ddqn_2.sh 3"
