#!/bin/bash
#SBATCH --array=1
#SBATCH --job-name=ddqn_2
#SBATCH --time=54:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodelist=mpu6
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --output/ssd6/san/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_10_24_23/%x_%a.out

#Parameter
default_config=$1 #inter arrival rate: 0-> 150, 1->240, 2->360, 3-> simplify instances
episodes=50000
batch_size=512
NN_size=512
learning_rate=0.000003
discount_factor=0.995


parent_dir='/ssd6/san/software/RL-approach-for-DHHSRP' #Change this
source_code="${parent_dir}/rl_dhhsrp/run/ddqn/training_ddqn.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/ddqn/"
. ${parent_dir}/.env/bin/activate

output_folder="output/ssd6/san/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_10_24_23"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"


mkdir $output_folder
echo " **** Training ddqn agent ..."
echo $source_code --output_folder $output_folder --config $default_config --episodes $episodes --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor '1>' $out_stream '2>' $err_stream
python3 $source_code --output_folder $output_folder --config $default_config --episodes $episodes --batch_size $batch_size --NN_size $NN_size --lr $learning_rate --discount_factor $discount_factor 1> $out_stream 2> $err_stream
echo " **** Done. :))"
