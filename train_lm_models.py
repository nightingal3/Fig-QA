import argparse
import logging
from typing import Optional
from glob import glob

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
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

def main(model_name: str, train_path: str, num_epochs: int, seed: int, use_cuda: bool, do_train: bool, do_eval: bool, cache_dir: str = "./lm_train_cache/") -> None:
    # Set up models, random seed, and logging
    model_names = {"gpt2": "gpt2", "neo-sm": "EleutherAI/gpt-neo-1.3B", "neo-lg": "EleutherAI/gpt-neo-2.7B"}
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
    training_args = { #TODO: make other training args customizable
        "output_dir": f"./lm_train_outputs/{model}/",
        "do_train": True,
        "prediction_loss_only": True,
        "per_device_batch_size": 8,
        "num_train_epochs": num_epochs,
        "block_size": tokenizer.model_max_length
        #"seed": seed
    }
    training_args = transformers.TrainingArguments(output_dir=f"./lm_train_outputs/{model_name}/", do_train=True, do_eval=False, 
    prediction_loss_only=True, num_train_epochs=num_epochs)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # Train the model
    if do_train:
        logger.info("=== Training the model ===")
        trainer.train()
        trainer.save_model("./lm_train_cache/")

    # Evaluate the model
    results = {}
    if do_eval:
        logger.info("=== Evaluating the model ===")
        acc = evaluate_model(model, tokenizer, trial_dataset["test"], acc_only=True)
        results["accuracy (dev)"] = acc

    with open(f"results_{model_name}.txt", "w") as writer:
        logger.info("=== Outputting results ===")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with the causal language modelling objective (GPT-*)")
    #TODO: add ability to load a pretrained model and just evaluate it
    parser.add_argument("model", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("-t", "--train_path", default="./lm_train_data/train.csv")
    parser.add_argument("-e", "--eval_path", default="./lm_train_data/dev.csv")
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-c", "--cuda", default=False, action="store_true")
    parser.add_argument("--num_epochs", default=3, type=int)
    args = parser.parse_args()
    main(args.model, args.train_path, args.num_epochs, args.seed, args.cuda, args.do_train, args.do_eval)