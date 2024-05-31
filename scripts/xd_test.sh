#!/bin/bash

#cd ../..

# custom config
DATA=//scratch-shared/promptlearning/coop_datasets/
TRAINER=CoPrompt

DATASET=$1
SEED=$2
EXP_NAME=$3

CFG=coprompt
SHOTS=16
LOADEP=$4
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

export MOMENTUM=0.0
export WEIGHT_STD=0.08
export MAX_EPOCH=32
export NUM_CONTEXT=2

export EXP_NAME_IPT=fullimnet
export COMMON_DIR_IPT=imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=//home/ataboadawarmer/data/fomo/output/${EXP_NAME_IPT}/train_base/${COMMON_DIR_IPT}
export DIR=//home/ataboadawarmer/data/fomo/output/${EXP_NAME}/test_${SUB}/${COMMON_DIR}

echo "Runing the first phase job and save the output to ${DIR}"

python train.py \
	--root ${DATA} \
	--seed ${SEED} \
	--trainer ${TRAINER} \
	--dataset-config-file configs/datasets/${DATASET}.yaml \
	--config-file configs/trainers/${CFG}.yaml \
	--output-dir ${DIR} \
	--model-dir ${MODEL_DIR} \
	--load-epoch ${LOADEP} \
	--eval-only
#	DATASET.NUM_SHOTS ${SHOTS} \
#	DATASET.SUBSAMPLE_CLASSES ${SUB}
