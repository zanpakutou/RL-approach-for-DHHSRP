#!/bin/bash
#SBATCH --array=1-80
#SBATCH --job-name=sba_
#SBATCH --time=30:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --nodelist=mpu5
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --output=/ssd6/san/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/log/%x_%a.out

parent_dir='/ssd6/san/software/RL-approach-for-DHHSRP'
#parent_dir="../../../"
source ${parent_dir}/.env/bin/activate
source_code="${parent_dir}/rl_dhhsrp/run/test/evaluate.py"
CURRENT_DIR="${parent_dir}/rl_dhhsrp/run/experiments/2022_10_26"
output_folder="${CURRENT_DIR}/"
#mkdir $output_folder
c=${SLURM_ARRAY_TASK_ID}
setting=$(sed -n "${c}p" listSetting.txt)

echo "Running DH, CH, SBA"
echo $setting
#python3 $source_code --obj $obj --nb_scenario $nb_scen --inter_arrival_rate $arr_rate --instance_type $type --output_folder $output_folder
python3 $source_code $setting