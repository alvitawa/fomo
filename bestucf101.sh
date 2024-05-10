exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh
dataset=ucf101
seed=1
CFG=coprompt

#data/fomo/output/CoPrompt_m0.0_wstd0.001_nctxt5_maxepoch8
export MOMENTUM=0.0
export WEIGHT_STD=0.001
export MAX_EPOCH=16
export NUM_CONTEXT=2
export exp_name=CoPrompt_m${MOMENTUM}_wstd${WEIGHT_STD}_nctxt${NUM_CONTEXT}_maxepoch${MAX_EPOCH}
export DIR=//home/ook/data/fomo/output/${exp_name}/train_base/${dataset}/shots_${SHOTS}/${trainer}/${CFG}/seed${seed}
#bash $train_bash $dataset $seed $exp_name
bash $test_bash $dataset $seed $exp_name $MAX_EPOCH