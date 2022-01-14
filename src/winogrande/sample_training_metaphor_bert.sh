#!/bin/sh
for LR in 2e-5
do
	for SEED in 0 1
	do
	echo "======================================================================="
	echo "========================= Metaphor BERT LR $LR seed $SEED ==========================="
	echo "======================================================================="
	CUDA_VISIBLE_DEVICES=2 python ./scripts/run_experiment.py \
	--model_type bert_mc \
	--model_name_or_path bert-large-uncased \
	--task_name winogrande \
	--do_eval \
	--do_lower_case \
	--data_dir ./data_metaphor \
	--max_seq_length 80 \
	--per_gpu_eval_batch_size 8 \
	--per_gpu_train_batch_size 8 \
	--learning_rate $LR \
	--num_train_epochs 7 \
	--output_dir ./output/models_metaphor_bert_$SEED/ \
	--do_train \
	--logging_steps 1002 \
	--save_steps 99999 \
	--seed $SEED \
	--data_cache_dir ./output/cache/ \
	--warmup_pct 0.1 \
	--evaluate_during_training \
	--overwrite_output_dir
	done
done
