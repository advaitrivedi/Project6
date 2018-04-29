#!/bin/bash
#
# 
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/home/advait/eclipse-workspace/TensoFlow/local_train_slim/tmp/alexnet-model

# Where the dataset is saved to.
DATASET_DIR=/home/advait/eclipse-workspace/TensoFlow/local_train_slim/tmp/flowers

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

# Run training.
#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=flowers \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=alexnet_v2 \
#  --preprocessing_name=alexnet_v2 \
#  --max_number_of_steps=20000 \
#  --batch_size=50 \
#  --learning_rate=0.01 \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \
#  --optimizer=sgd \
#  --learning_rate_decay_type=fixed \
#  --weight_decay=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2
