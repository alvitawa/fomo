exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh
seed=1
dataset=dtd
CFG=coprompt

#data/fomo/output/CoPrompt_m0.0_wstd0.001_nctxt5_maxepoch8
export MOMENTUM=0.0
export WEIGHT_STD=0.012
export MAX_EPOCH=16
#export MAX_EPOCH=2
export NUM_CONTEXT=2
export exp_name=transferds
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/test_new/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"

export NOSHARED=1
export exp_name=transferds_noshared
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/test_new/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
unset NOSHARED

export NOSELECT=1
export exp_name=transferds_noselect
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/test_new/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
unset NOSELECT

export NOFILTER=1
export exp_name=transferds_nofilter
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/test_new/${dataset}/shots_16/CoPrompt/seed${seed}"
mkdir -p "$output_dir"
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH > "${output_dir}/output.log" 2>&1
cat "${output_dir}/output.log"
unset NOFILTER
