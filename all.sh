exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Run all training in parallel, then wait for all to finish
for seed in 1 2 3; do
  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
    bash $train_bash $dataset $seed $exp_name &
  done
done
wait

# Run all tests in parallel, then wait for all to finish
for seed in 1 2 3; do
  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
    test_arg=8
    [ "$dataset" = "food101" ] && test_arg=5 # Special case for food101
    bash $test_bash $dataset $seed $exp_name $test_arg &
  done
done
wait
