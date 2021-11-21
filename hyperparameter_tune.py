from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import argparse
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

from train_lm_models import training_setup
from gpt_score import model_init

dummy_init = lambda m: m

def optimize(model_name: str, use_cuda: bool, contrastive_train: bool, seed: int, lr: float, num_epochs: int, train_path: str) -> None:
    model_names = {"gpt2": "gpt2", "gpt-neo-sm": "EleutherAI/gpt-neo-1.3B", "gpt-neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_string = model_names[model_name]
    model, tokenizer = model_init(model_string, use_cuda)

    trainer = training_setup(model, tokenizer, model_string, seed, lr, num_epochs, train_path, contrastive_train=contrastive_train, is_hyperparam_opt=True)

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="ray",
        search_alg=HyperOptSearch(metric="objective", mode="min"),
        scheduler=ASHAScheduler(metric="objective", mode="min"),
        hp_space= lambda _ : {
            "num_train_epochs": tune.quniform(3, 8, 1),
            "learning_rate": tune.quniform(1e-5, 7e-5, 1e-6)
        }
    )

    return best_trial



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tune the hyperparameters for a model")
    parser.add_argument("model", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("-t", "--train_path", default="./lm_train_data/train.txt")

    args = parser.parse_args()

    best_trial = optimize(args.model, args.cuda, args.contrastive, args.seed, args.lr, args.num_epochs, args.train_path)
    print(best_trial)
