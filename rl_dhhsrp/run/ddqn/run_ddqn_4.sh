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

parent_dir='/home/quy/Repos/RL_DHHSRP' #Change this
. ${parent_dir}/env/bin/activate
source_code="${parent_dir}/rl_dhhsrp/run/ddqn/training_ddqn.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/ddqn/"

#Parameter
default_config=3 #inter arrival rate: 0-> 150, 1->240, 2->360, 3-> simplify instances
time_steps=50000
batch_size=512
NN_size=512
learning_rate=1e-6
discount_factor=0.99

output_folder="${CURRENT_DIR}/${default_config}-${time_steps}-${batch_size}-${NN_size}-${learning_rate}-${discount_factor}"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"

mkdir $output_folder
echo " **** Training ddqn agent ..."
echo $source_code --output_folder $output_folder --config $default_config --timesteps $time_steps --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor 

python3 $source_code --output_folder $output_folder --config $default_config --timesteps $time_steps --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor 1> $out_stream 2> $err_stream
echo " **** Done. :))"
