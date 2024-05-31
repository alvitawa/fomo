exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh
seed=1
CFG=coprompt

for dataset in fgvc_aircraft caltech101 oxford_pets stanford_cars food101 sun397 ucf101 dtd oxford_flowers eurosat; do
  #data/fomo/output/CoPrompt_m0.0_wstd0.001_nctxt5_maxepoch8
  export MOMENTUM=0.0
  export WEIGHT_STD=0.001
  export MAX_EPOCH=1

  export NUM_CONTEXT=2
  export exp_name=transferds_real
  export DIR=//home/ook/data/fomo/output/${exp_name}/train_base/${dataset}/shots_${SHOTS}/${trainer}/${CFG}/seed${seed}
  export PROMPT="conveniently photograph"
  echo "Running ${exp_name} on ${dataset}"
  bash $train_bash $dataset $seed $exp_name
  bash $test_bash $dataset $seed $exp_name $MAX_EPOCH

  export NUM_CONTEXT=4
  export exp_name=transferds_base
  export DIR=//home/ook/data/fomo/output/${exp_name}/train_base/${dataset}/shots_${SHOTS}/${trainer}/${CFG}/seed${seed}
  export PROMPT="a photo of a"
  echo "Running ${exp_name} on ${dataset}"
  bash $train_bash $dataset $seed $exp_name
  bash $test_bash $dataset $seed $exp_name $MAX_EPOCH
done