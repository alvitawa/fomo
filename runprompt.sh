exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh
dataset=eurosat
seed=1
CFG=coprompt

#export PROMPT="increased underdog gihc"

#data/fomo/output/CoPrompt_m0.0_wstd0.001_nctxt5_maxepoch8
export MOMENTUM=0.0
export WEIGHT_STD=0.012
export MAX_EPOCH=16
#export MAX_EPOCH=2
export NUM_CONTEXT=4
export exp_name=transferds
bash $train_bash $dataset $seed $exp_name
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH