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
instance_type=uniform
arr_rate = 240
time_steps=2500000
batch_size=256 #only for dqn
NN_size=512
learning_rate=0.00001
discount_factor=0.99
alg=DQN #DQN, A2C, PPO
obj=patient

output_folder="${CURRENT_DIR}/${instance_type}-${time_steps}-${batch_size}-${NN_size}-${learning_rate}-${discount_factor}-${obj}-${alg}"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"
ls $CURRENT_DIR
mkdir   $output_folder
echo    $output_folder
echo " **** Training SB agent ..."
echo $source_code --output_folder $output_folder --instance_type $instance_type --timesteps $time_steps --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor --obj $obj --alg $alg
python3 $source_code --output_folder $output_folder --instance_type $instance_type --timesteps $time_steps --instance_type $instance_type --arr_rate $arr_rate --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor --obj $obj --alg $alg 1> $out_stream 2> $err_stream
echo " **** Done."
