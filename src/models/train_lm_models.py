import argparse
import logging
from typing import Optional
from glob import glob
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
import pickle

import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    EarlyStoppingCallback
)
from torch.utils.data import ConcatDataset
import pdb

from gpt_score import model_init, evaluate_model

logger = logging.getLogger(__name__)

def main(model_name: str, prompt: str, train_path: str, eval_path: str, contrastive_train: bool, contrastive_train_lambd: float, num_epochs: int, seed: int, lr: int, use_cuda: bool, dont_train: bool, dont_eval: bool, out_path: str, cache_dir: str = "./lm_train_cache/", prefix_prompt: int = 0, batch_size: int = 8, log_history: bool = False, deepspeed: bool = False, early_stopping: bool = False) -> None:
    # Set up models, random seed, and logging
    model_names = {"gpt2": "gpt2", "gpt-neo-sm": "EleutherAI/gpt-neo-1.3B", "gpt-neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_id = model_names[model_name]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", {"model": model_name, "train path": train_path, "num epochs": num_epochs, "seed": seed, "cuda": use_cuda, "cache dir": cache_dir, "deepspeed": deepspeed, "early stopping": early_stopping})


    if deepspeed and not use_cuda:
        logger.info("You must have GPUs to use deepspeed. Turning cuda flag on...")
        use_cuda = True

    model, tokenizer = model_init(model_id, use_cuda, fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    #model.resize_token_embeddings(len(tokenizer))
    set_seed(seed)
    
    # load datasets and initialize trainer
    train_dataset = (
        get_dataset(train_path, tokenizer=tokenizer, cache_dir=cache_dir)
    )
    eval_dataset = (
        get_dataset(eval_path, tokenizer=tokenizer, cache_dir=cache_dir)
    )

    eval_df = pd.read_csv("./data/filtered/dev.csv")
    eval_df["label"] = eval_df["labels"]
    test_df = pd.read_csv("./data/filtered/dev.csv")
    test_df["label"] = test_df["labels"]

    data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
    no_cuda = not use_cuda

    default_arguments = {
        "output_dir": f"./lm_train_outputs/{model_name}_{seed}/",
        "do_train": True,
        "prediction_loss_only": False, 
        "num_train_epochs": num_epochs,
        "seed": seed,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "no_cuda": no_cuda
    }
    
    if deepspeed:
        default_arguments["deepspeed"] = "deepspeed_config.json"
    if not contrastive_train:
        default_arguments["per_device_train_batch_size"] = batch_size
        default_arguments["per_device_eval_batch_size"] = batch_size

    else:
        default_arguments["per_device_train_batch_size"] = 2
        
    if log_history:
        default_arguments["evaluation_strategy"] = "steps"
        default_arguments["eval_steps"] = 100
    if early_stopping:
        default_arguments["evaluation_strategy"] = "epoch"
        default_arguments["load_best_model_at_end"] = True
        default_arguments["metric_for_best_model"] = "eval_loss"
        default_arguments["save_strategy"] = "epoch"

    training_args = transformers.TrainingArguments(**default_arguments)
    
    if early_stopping:
        trainer = Trainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
    elif not contrastive_train:
        #tokenizer.pad_token = tokenizer.eos_token
        #dummy_init = make_dummy(model_id)
        trainer = Trainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            #model_init=dummy_init,
            compute_metrics=compute_metrics
        )
    else:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.set_lambd(contrastive_train_lambd)

    # Train the model
    if not dont_train:
        logger.info("=== Training the model ===")
        trainer.train()
        trainer.save_model("./lm_train_cache/")
        if log_history:
            log_file = f"{model_name}_epochs_{num_epochs}_eval_loss.p"
            with open(log_file, "wb") as f:
                pickle.dump(trainer.state.log_history, f)
    
    # Evaluate the model
    results = {}
    if not dont_eval: #Note: for hyperparameter tuning we do it by loss on 
        model.eval()
        logger.info("=== Evaluating the model ===")
        eval_output = trainer.evaluate()
        eval_loss = eval_output["eval_loss"]
        results["eval_loss"] = eval_loss
        
        acc_test, out_df_test, preds_test, labels_test = evaluate_model(model, tokenizer, test_df.to_dict(orient="records"), use_cuda=use_cuda, return_acc=True, middle_phrase=prompt, use_prefix=prefix_prompt)
        acc_dev, out_df_dev, preds_dev, labels_dev = evaluate_model(model, tokenizer, eval_df.to_dict(orient="records"), use_cuda=use_cuda, return_acc=True, middle_phrase=prompt, use_prefix=prefix_prompt)
        results["accuracy (test)"] = acc_test
        results["accuracy (dev)"] = acc_dev
        results["preds"] = preds_test
        results["labels"] = labels_test


    if out_path is not None:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        with open(f"{out_path}/results_{model_name}.txt", "w") as writer:
            logger.info("=== Outputting results ===")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

        out_df_test.to_csv(f"{out_path}/prob_{model_name}_{seed}.csv", index=False)
    
    return results

def training_setup(model, tokenizer, model_name, seed, lr, num_epochs, train_path, eval_path, contrastive_train=False, contrast_lambd=1, is_hyperparam_opt=False, cuda=True, deepspeed=False, batch_size=8) -> Trainer:
    # load datasets and initialize trainer
    train_dataset = (
        get_dataset(train_path, tokenizer=tokenizer)
    )
    eval_dataset = (
        get_dataset(eval_path, tokenizer=tokenizer)
    )

    data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
    set_seed(seed)

    default_train_args = {
        "output_dir": f"./lm_train_outputs/{model_name}_{seed}/",
        "do_train": True,
        "do_eval": False,
        "prediction_loss_only": True,
        "seed": seed,
        "num_train_epochs": num_epochs, 
        "learning_rate": lr,
        "no_cuda": not cuda,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size
    }

    if contrastive_train:
        default_train_args["per_device_train_batch_size"] = 2 
        training_args = transformers.TrainingArguments(output_dir=f"./lm_train_outputs/{model_name}_{seed}/", do_train=True, do_eval=False, 
        prediction_loss_only=True, num_train_epochs=num_epochs, seed=seed,learning_rate=lr, per_device_train_batch_size=2)
    elif is_hyperparam_opt:
        default_train_args["evaluation_strategy"] = "steps"
        default_train_args["eval_steps"] = 500
        default_train_args["disable_tqdm"] = True
    if deepspeed == True:
        default_train_args["deepspeed"] = "./deepspeed_config.json"

    training_args = transformers.TrainingArguments(**default_train_args)

    
    if is_hyperparam_opt:
        tokenizer.pad_token = tokenizer.eos_token
        dummy_init = make_dummy(model_name)
        trainer = Trainer(
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=dummy_init,
            compute_metrics=compute_metrics
        )
    elif contrastive_train:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.set_lambd(contrast_lambd)
    else: 
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
    return trainer

# This is adapted from the huggingface LM training example here: https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py
def get_dataset(
    train_data_file: str,
    tokenizer: PreTrainedTokenizer,
    line_by_line: bool = True,
    evaluate: bool = False,
    eval_data_file: str = None,
    cache_dir: Optional[str] = None,
):
    def _dataset(file_path, ref_path=None):
        if line_by_line:
            if ref_path is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError("You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=tokenizer.model_max_length,
                    ref_path=ref_path,
                )

            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=tokenizer.model_max_length)

    if evaluate:
        return _dataset(eval_data_file)
    else:
        return _dataset(train_data_file)

def make_dummy(model_id):
    def dummy_init():
        if model_id == "gpt2":
            return GPT2LMHeadModel.from_pretrained("gpt2", return_dict=True)
        elif "gpt-neo" in model_id:
            return GPTNeoForCausalLM.from_pretrained(model_id, return_dict=True)
    return dummy_init

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    acc = len(np.where(predictions == labels)[0])/len(labels)
    return {"acc": acc}

class ContrastiveTrainer(Trainer):
    def set_lambd(self, lambd):
        self.lambd = lambd

    def compute_loss(self, model, inputs, return_outputs=False):
        # Assumes batch size of 2!
        if inputs["labels"].shape[0] % 2 != 0:
            raise ValueError("Batch size must be a multiple of 2")
        
        correct_inputs = {"input_ids": torch.stack([row for i, row in enumerate(inputs["input_ids"]) if i % 2 == 0]),
        "attention_mask": torch.stack([row for i, row in enumerate(inputs["attention_mask"]) if i % 2 == 0]), 
        "labels":  torch.stack([row for i, row in enumerate(inputs["labels"]) if i % 2 == 0])}
        wrong_inputs = {"input_ids": torch.stack([row for i, row in enumerate(inputs["input_ids"]) if i % 2 == 1]),
        "attention_mask": torch.stack([row for i, row in enumerate(inputs["attention_mask"]) if i % 2 == 1]), 
        "labels":  torch.stack([row for i, row in enumerate(inputs["labels"]) if i % 2 == 1])}

        outputs = model(**inputs)

        correct_outputs = model(**correct_inputs)
        correct_loss = correct_outputs.get('loss')

        wrong_outputs = model(**wrong_inputs)
        wrong_loss = wrong_outputs.get("loss")

        # Good = when the loss for the correct item is much lower than loss for wrong item
        # loss should be negative (good) when wrong loss > correct loss
        #lambd = self.lambd if self.lambd else 1
        lambd = 0.2
        relative_score = correct_loss - lambd * (wrong_loss + correct_loss)
        loss = -relative_score

        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with the causal language modelling objective (GPT-*)")
    #TODO: add ability to load a pretrained model and just evaluate it
    parser.add_argument("model", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--dont_train", action="store_true", default=False)
    parser.add_argument("--dont_eval", action="store_true", default=False)
    parser.add_argument("-t", "--train_path", default="./data/lm_train_data/train.txt")
    parser.add_argument("-e", "--eval_path", default="./data/lm_train_data/dev.txt")
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-c", "--cuda", default=False, action="store_true")
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--middle_phrase", default="")
    parser.add_argument("--prefix", default=0)
    parser.add_argument("--contrastive", default=False, action="store_true")
    parser.add_argument("--contrast_lambd", type=float, default=1)
    parser.add_argument("--log_history", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--out_path")
    parser.add_argument("--early_stopping", action="store_true", default=False)
    args = parser.parse_args()

    contrast_path = "contrast/" if args.contrastive else ""

    if args.out_path is not None:
        out_path = args.out_path
    else:
        out_path = f"./experiments/{contrast_path}{args.model}/epochs_{args.num_epochs}_{args.learning_rate}_/seed_{args.seed}"

    if args.learning_rate is None:
        learning_rate = 5e-5
    else:
        learning_rate = args.learning_rate
        
    if args.model != "gpt2" and args.cuda:
        deepspeed = True
    else:
        deepspeed = args.deepspeed

    main(args.model, args.middle_phrase, args.train_path, args.eval_path, args.contrastive, args.contrast_lambd, args.num_epochs, args.seed, learning_rate, args.cuda, args.dont_train, args.dont_eval, out_path, prefix_prompt=args.prefix, log_history=args.log_history, deepspeed=deepspeed, early_stopping=args.early_stopping)
