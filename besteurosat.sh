exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh
dataset=eurosat
#seed=1
CFG=coprompt

#data/fomo/output/CoPrompt_m0.0_wstd0.001_nctxt5_maxepoch8
export MOMENTUM=0.0
export WEIGHT_STD=0.012
export MAX_EPOCH=16
export NUM_CONTEXT=4
export exp_name=CoPrompt_m${MOMENTUM}_wstd${WEIGHT_STD}_nctxt${NUM_CONTEXT}_maxepoch${MAX_EPOCH}
export DIR=//home/ook/data/fomo/output/${exp_name}/train_base/${dataset}/shots_${SHOTS}/${trainer}/${CFG}/seed${seed}
#bash $train_bash $dataset $seed $exp_name
seed=3
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
echo "Runing the first phase job and save the output to ${output_dir}"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1 &

seed=2
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
echo "Runing the first phase job and save the output to ${output_dir}"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1 &

seed=1
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
echo "Runing the first phase job and save the output to ${output_dir}"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1 &
