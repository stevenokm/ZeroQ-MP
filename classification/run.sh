#!/bin/bash -e
# for MODEL in resnet18 resnet50 inceptionv3 mobilenetv2_w1 shufflenet_g1_w1 sqnxt23_w2; do
# for MODEL in resnet18 resnet50 mobilenetv2_w1 shufflenet_g1_w1; do
EXPR_TIME=10
TEST_BATCH_SIZE=512
BATCH_SIZE=64
DATASET=imagenet
DATA_SOURCE=train

echo Using $DATASET dataset with $DATA_SOURCE data source.
echo Batch size is $BATCH_SIZE, test batch size is $TEST_BATCH_SIZE.
echo Testing $EXPR_TIME times for each model.

for MODEL in resnet18; do
    echo Testing $MODEL ...
    for i in $(seq 1 $EXPR_TIME); do
        python uniform_test.py \
            --dataset=$DATASET \
            --model=$MODEL \
            --batch_size=$BATCH_SIZE \
            --test_batch_size=$TEST_BATCH_SIZE \
            --data-source=$DATA_SOURCE
    done
    echo Testing $MODEL with per-channel quant. ...
    for i in $(seq 1 $EXPR_TIME); do
        python uniform_test.py \
            --dataset=$DATASET \
            --model=$MODEL \
            --batch_size=$BATCH_SIZE \
            --test_batch_size=$TEST_BATCH_SIZE \
            --data-source=$DATA_SOURCE \
            --per_channel
    done
    for TILE_SIZE in 320 256 192 128 64 32 16; do
        echo Testing $MODEL with per-tile quant. \(tile size $TILE_SIZE ...\)
        for i in $(seq 1 $EXPR_TIME); do
            python uniform_test.py \
                --dataset=$DATASET \
                --model=$MODEL \
                --batch_size=$BATCH_SIZE \
                --test_batch_size=$TEST_BATCH_SIZE \
                --data-source=$DATA_SOURCE \
                --tile_size=$TILE_SIZE
        done
    done
done
