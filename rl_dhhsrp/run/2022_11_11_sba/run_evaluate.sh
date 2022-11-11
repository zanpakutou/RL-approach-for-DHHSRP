#!/bin/bash
#SBATCH --array=1-48
#SBATCH --job-name=sba
#SBATCH --time=160:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=def-roussea5
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_11_11_sba/log/%x_%a.out

parent_dir='/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP'
#parent_dir="../../../"
source ${parent_dir}/.env3.7/bin/activate
source_code="${parent_dir}/rl_dhhsrp/run/test/evaluate.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/2022_11_11_sba"
output_folder="${CURRENT_DIR}/sba/"
#mkdir $output_folder
c=${SLURM_ARRAY_TASK_ID}
setting=$(sed -n "${c}p" listSettingSBA.txt)

echo "Running DH, CH, SBA"
echo $setting
#python3 $source_code --obj $obj --nb_scenario $nb_scen --inter_arrival_rate $arr_rate --instance_type $type --output_folder $output_folder
python3 $source_code $setting
