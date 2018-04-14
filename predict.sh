#!/usr/bin/env bash

source activate py36_1

MODEL_PATH=/media/yanpan/7D4CF1590195F939/Projects/tf-pose-model/mytrousers_prof/tf-pose-1-trousers
MODEL_NAME=$MODEL_PATH/cmu_batch:32_lr:0.0001_gpus:1_368x368_/model-27000

python ./src/run_checkpoint.py --model=$MODEL_NAME --pb_path=$MODEL_PATH/graph.pb

python -m tensorflow.python.tools.freeze_graph \
  --input_graph=$MODEL_PATH/graph.pb \
  --output_graph=$MODEL_PATH/graph_freeze.pb \
  --input_checkpoint=$MODEL_NAME \
  --output_node_names="Openpose/concat_stage7"

# TODO: graph_opt.pb
# tensorflow brazel

#python ./src/run.py --model $MODEL_PATH/graph_freeze.pb \
#                    --resolution 368x368 \
#                    --image /media/yanpan/7D4CF1590195F939/Projects/fashionai/mytry/val2017/0faaa01d3ae07334448b8fe7643febc8.jpg
