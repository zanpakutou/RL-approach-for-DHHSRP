#!/bin/bash
#SBATCH --array=1-2
#SBATCH --job-name=ddqn_2
#SBATCH --time=54:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodelist=mpu6
#SBATCH --mail-user=phamtusan@gmail.com
#SBATCH --output/ssd6/san/software/RL-approach-for-DHHSRP/rl_dhhsrp/run/2022_10_24_23_test/%x_%a.out

source /home/tusan/projects/def-roussea5/tusan/software/penv/bin/activate

BASEDIR=/home/tusan/projects/def-roussea5/tusan/exp/2021_Oct
source_code=$BASEDIR/RTSP-scheduling/main.py

CURDIR=$BASEDIR/7linacs/lambda10.1
datadir=$CURDIR/data
folder_result=$CURDIR/output/results
folder_solution=$CURDIR/output/solutions_static

# for static - all instances
declare -a solver=("static")
timeout=54000


c=${SLURM_ARRAY_TASK_ID}
insname=$(sed -n "${c}p" listInstances.txt)
echo $insname




for s in "${solver[@]}"
do
	echo "solving $s - instance $insname"
	res_file="$folder_solution/$insname-$s.sol"
	if [ -f $res_file ]; then
		echo "File $res_file exists, skip."
	else
		python3 $source_code --mode server --solver $s --timeout $timeout --ins_path $datadir --ins_name $insname --solution_path $folder_solution --result_path $folder_result
	fi
done

