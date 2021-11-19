import argparse
import logging
from typing import Optional
from glob import glob
from pathlib import Path
import os
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import ConcatDataset
from sample_metaphors import trial_dataset
import pdb

from gpt_score import model_init, evaluate_model

logger = logging.getLogger(__name__)

def main(model_name: str, prompt: str, train_path: str, contrastive_train: bool, num_epochs: int, seed: int, lr: int, use_cuda: bool, dont_train: bool, dont_eval: bool, out_path: str, cache_dir: str = "./lm_train_cache/") -> None:
    # Set up models, random seed, and logging
    model_names = {"gpt2": "gpt2", "gpt-neo-sm": "EleutherAI/gpt-neo-1.3B", "gpt-neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_id = model_names[model_name]
    model, tokenizer = model_init(model_id, use_cuda, fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    #model.resize_token_embeddings(len(tokenizer))
    set_seed(seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", {"model": model_name, "train path": train_path, "num epochs": num_epochs, "seed": seed, "cuda": use_cuda, "cache dir": cache_dir})

    # load datasets and initialize trainer
    train_dataset = (
        get_dataset(train_path, tokenizer=tokenizer, cache_dir=cache_dir)
    )

    data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
    if not contrastive_train:
        training_args = transformers.TrainingArguments(output_dir=f"./lm_train_outputs/{model_name}_{seed}/", do_train=True, do_eval=False, 
        prediction_loss_only=True, num_train_epochs=num_epochs, seed=seed,learning_rate=lr)
    else:
        training_args = transformers.TrainingArguments(output_dir=f"./lm_train_outputs/{model_name}_{seed}/", do_train=True, do_eval=False, 
        prediction_loss_only=True, num_train_epochs=num_epochs, seed=seed,learning_rate=lr, per_device_train_batch_size=2)

    if not contrastive_train:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )
    else:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )

    # Train the model
    if not dont_train:
        logger.info("=== Training the model ===")
        trainer.train()
        trainer.save_model("./lm_train_cache/")

    # Evaluate the model
    results = {}
    if not dont_eval:
        model.eval()
        logger.info("=== Evaluating the model ===")
        acc, out_df, preds, labels = evaluate_model(model, tokenizer, trial_dataset["test"], use_cuda=use_cuda, return_acc=True, middle_phrase=prompt)
        results["accuracy (dev)"] = acc
        results["preds"] = preds
        results["labels"] = labels

    Path(out_path).mkdir(parents=True, exist_ok=True)
    with open(f"{out_path}/results_{model_name}.txt", "w") as writer:
        logger.info("=== Outputting results ===")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    out_df.to_csv(f"{out_path}/prob_{model_name}_{seed}.csv", index=False)

    return results

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

class ContrastiveTrainer(Trainer):
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

        loss = wrong_loss - correct_loss

        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with the causal language modelling objective (GPT-*)")
    #TODO: add ability to load a pretrained model and just evaluate it
    parser.add_argument("model", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--dont_train", action="store_true", default=False)
    parser.add_argument("--dont_eval", action="store_true", default=False)
    parser.add_argument("-t", "--train_path", default="./lm_train_data/train.txt")
    parser.add_argument("-e", "--eval_path", default="./lm_train_data/dev.txt")
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-c", "--cuda", default=False, action="store_true")
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--middle_phrase", default="")
    parser.add_argument("--contrastive", default=False, action="store_true")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    if args.out_path is not None:
        out_path = args.out_path
    else:
        out_path = f"./experiments/{args.model}/epochs_{args.num_epochs}_{args.learning_rate}_/seed_{args.seed}"

    if args.learning_rate is None:
        learning_rate = 5e-5
    else:
        learning_rate = args.learning_rate

    main(args.model, args.middle_phrase, args.train_path, args.contrastive, args.num_epochs, args.seed, learning_rate, args.cuda, args.dont_train, args.dont_eval, out_path)