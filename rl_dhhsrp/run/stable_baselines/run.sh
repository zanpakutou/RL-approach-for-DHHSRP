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
parent_dir='../../../' #Change this
. ${parent_dir}/env/bin/activate
source_code="${parent_dir}/rl_dhhsrp/run/stable_baselines/training.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/stable_baselines/"

#Parameter
instance_type=1 #inter arrival rate: 0-> 150, 1->240, 2->360, 3-> simplify instances
time_steps=5000000
batch_size=512 #only for dqn
NN_size=216
learning_rate=0.00001
discount_factor=0.995
alg=DQN #DQN, A2C, PPO
obj=patient #visit

output_folder="${CURRENT_DIR}/${instance_type}-${time_steps}-${batch_size}-${NN_size}-${learning_rate}-${discount_factor}"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"
ls $CURRENT_DIR
mkdir $output_folder
echo " **** Training SB agent ..."
echo $source_code --output_folder $output_folder --instance_type $instance_type --timesteps $time_steps --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor --obj $obj
python3 $source_code --output_folder $output_folder --instance_type $instance_type --timesteps $time_steps --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor --obj $obj 1> $out_stream #2> $err_stream
echo " **** Done. :))"
