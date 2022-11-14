#!/bin/bash
#SBATCH --array=1-9
#SBATCH --job-name=ppo_dhhsrp
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=def-roussea5
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_11_14_PPO/log/%x_%a.out

parent_dir='/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP'
#parent_dir='../../../'
source_code="${parent_dir}/rl_dhhsrp/run/stable_baselines/training.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/2022_11_14_PPO/"
source ${parent_dir}/.env3.7/bin/activate

c=${SLURM_ARRAY_TASK_ID}
setting=$(sed -n "${c}p" listSettingDDQN.txt)

#Read the parameter
IFS=' '
read -a params <<< "$setting"
timesteps=${params[0]}
batch_size=${params[1]}
discount_factor=${params[2]}
instance_type=${params[3]}
arr_rate=${params[4]}
obj=${params[5]}
lr=${params[6]}
cap_heur=${params[7]}
nb_nurse=${params[8]}

output_folder="${CURRENT_DIR}/ppo/${instance_type}-${arr_rate}-${nb_nurse}-${obj}-${cap_heur}-${timesteps}-${batch_size}-${discount_factor}/"
out_stream="${output_folder}/out.txt"
err_stream="${output_folder}/err.txt"

echo $output_folder
mkdir -p $output_folder
echo " **** Training ddqn agent ..."
echo $source_code --timesteps $timesteps --batch_size $batch_size --discount_factor $discount_factor --output_folder ${output_folder} --lr ${lr} --obj ${obj} --instance_type ${instance_type} --arr_rate ${arr_rate} --nb_nurse ${nb_nurse} --cap_heur ${cap_heur} --alg PPO
python3 $source_code --timesteps $timesteps --batch_size $batch_size --discount_factor $discount_factor --output_folder ${output_folder} --lr ${lr} --obj ${obj} --instance_type ${instance_type} --arr_rate ${arr_rate} --nb_nurse ${nb_nurse} --cap_heur ${cap_heur} --alg PPO 1> $out_stream 2>$err_stream
echo " **** Done."
