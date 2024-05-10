#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu

#Execute program located in $HOME
#$HOME/my_serial_program

#cd $HOME/workspace || exit


exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Run all training in parallel, then wait for all to finish
for seed in 1; do
  for momentum in 0.0 0.9; do
    export MOMENTUM=$momentum
    for wstd in 0.001 0.012 0.08; do
      export WEIGHT_STD=$wstd
      for max_epoch in 8 16; do
        export MAX_EPOCH=$max_epoch
        for nctxt in 1 2 3 4; do
          export NUM_CONTEXT=$nctxt
          for dataset in oxford_flowers eurosat; do
            export exp_name=CoPrompt_m${momentum}_wstd${wstd}_nctxt${nctxt}_maxepoch${max_epoch}
            bash $train_bash $dataset $seed $exp_name &
          done
        done
        wait
      done
    done
  done
done





#  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
#    bash $train_bash $dataset $seed $exp_name &
#  done
#done
#wait

## Run all tests in parallel, then wait for all to finish
#for seed in 1 2 3; do
#  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
#    test_arg=8
#    [ "$dataset" = "food101" ] && test_arg=5 # Special case for food101
#    bash $test_bash $dataset $seed $exp_name $test_arg &
#  done
#done
#wait

for seed in 1; do
  for momentum in 0.0 0.9; do
    export MOMENTUM=$momentum
    for wstd in 0.001 0.012 0.08; do
      export WEIGHT_STD=$wstd
      for max_epoch in 8 16; do
        export MAX_EPOCH=$max_epoch
        for nctxt in 1 2 3 4; do
          export NUM_CONTEXT=$nctxt
          for dataset in oxford_flowers eurosat; do
            export exp_name=CoPrompt_m${momentum}_wstd${wstd}_nctxt${nctxt}_maxepoch${max_epoch}
            bash $test_bash $dataset $seed $exp_name 8 &
          done
        done
        wait
      done
    done
  done
done