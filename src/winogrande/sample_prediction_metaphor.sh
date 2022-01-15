#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type roberta_mc \
--model_name_or_path ./output/models_metaphor_roberta \
--task_name winogrande \
--do_eval \
--do_lower_case \
--data_dir ./data_metaphor_test_set \
--max_seq_length 80 \
--per_gpu_eval_batch_size 8 \
--output_dir ./output/models_metaphor_roberta/ \
--data_cache_dir ./output/cache/ 

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type bert_mc \
--model_name_or_path ./output/models_metaphor_bert \
--task_name winogrande \
--do_eval \
--do_lower_case \
--data_dir ./data_metaphor_test_set \
--max_seq_length 80 \
--per_gpu_eval_batch_size 8 \
--output_dir ./output/models_metaphor_bert/ \
--data_cache_dir ./output/cache/ 