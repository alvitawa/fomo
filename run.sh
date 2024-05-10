

source load.sh

# set XTRACE environment variable to pass along to the job
export XTRACE=0



srun --nodes=1 --ntasks=1 --cpus-per-task=18 --gpus=1 --partition=gpu --time=1:00:00 --cpus-per-task=18 "$@"

