#!/bin/bash
#SBATCH --array=1-112
#SBATCH --job-name=ddqn
#SBATCH --time=144:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --account=def-roussea5
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_10_27/log/%x_%a.out

parent_dir='/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP' #Change this
#parent_dir='../../../'
source_code="${parent_dir}/rl_dhhsrp/run/ddqn/training_ddqn.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/2022_10_27/"
source ${parent_dir}/.env/bin/activate

c=${SLURM_ARRAY_TASK_ID}
setting=$(sed -n "${c}p" listSettingDDQN.txt)

#Read the parameter
IFS=' '
read -a params <<< "$setting"
episodes=${params[0]}
batch_size=${params[1]}
NN_size=${params[2]}
discount_factor=${params[3]}
instance_type=${params[4]}
arr_rate=${params[5]}
obj=${params[6]}
lr=${params[7]}
transition_type=${params[8]}

output_folder="${CURRENT_DIR}/ddqn/${instance_type}-${arr_rate}-${obj}-${transition_type}-${episodes}-${batch_size}-${NN_size}-${discount_factor}/"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"

echo $output_folder
mkdir -p $output_folder
echo " **** Training ddqn agent ..."
echo $source_code --episodes $episodes --batch_size $batch_size --NN_size $NN_size --discount_factor $discount_factor --output_folder ${output_folder} --config 0 --lr ${lr} --obj ${obj} --arr_rate ${arr_rate} --transition_type ${transition_type}

python3 $source_code --episodes $episodes --batch_size $batch_size --NN_size $NN_size --discount_factor $discount_factor --output_folder ${output_folder} --config 0 --lr ${lr} --obj ${obj} --arr_rate ${arr_rate} --transition_type ${transition_type} 1> $out_stream 2> $err_stream
echo " **** Done."