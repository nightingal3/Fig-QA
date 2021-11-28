#from ray import tune
#from ray.tune.suggest.hyperopt import HyperOptSearch
#from ray.tune.schedulers import ASHAScheduler
import argparse
import pdb
import pickle

from train_lm_models import training_setup, main
from gpt_score import model_init

def optimize(model_name: str, use_cuda: bool, contrastive_train: bool, seed: int, lr: float, num_epochs: int, train_path: str, eval_path: str) -> None:
    model_names = {"gpt2": "gpt2", "gpt-neo-sm": "EleutherAI/gpt-neo-1.3B", "gpt-neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_string = model_names[model_name]
    model, tokenizer = model_init(model_string, use_cuda)
    batch_size = 8

    trainer = training_setup(model, tokenizer, model_string, seed, lr, num_epochs, train_path, eval_path, contrastive_train=contrastive_train, is_hyperparam_opt=True, deepspeed=False, batch_size=batch_size)

    if contrastive_train:
        hp_space = lambda _ : {
            "num_train_epochs": tune.quniform(3, 8, 1),
            "learning_rate": tune.quniform(1e-5, 7e-5, 1e-6),
            "lambd": tune.quniform(0.1, 1, 0.1)
        }
    else:
        hp_space = lambda _ : {
            "num_train_epochs": tune.quniform(3, 5, 1),
            "learning_rate": tune.quniform(1e-5, 5e-5, 1e-6)
        }

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="ray",
        search_alg=HyperOptSearch(metric="objective", mode="min"),
        scheduler=ASHAScheduler(metric="objective", mode="min"),
        hp_space=hp_space
    )

    return best_trial

def optimize_minimal(model_name: str, use_cuda: bool, contrastive_train: bool, seed: int, train_path: str, eval_path: str) -> tuple:
    if contrastive_train:
        lambd_lst = [0.2 * i for i in range(1, 6)]
        batch_size = 2

        trials = {}
        best_eval_loss = float("inf")
        best_trial = None

        for lambd in lambd_lst:
            #trainer = training_setup(model, tokenizer, model_string, seed, lr, num_epochs, train_path, contrastive_train=contrastive_train, is_hyperparam_opt=False)
            results = main(model_name, "", train_path, eval_path, contrastive_train, lambd, 3, seed, 5e-5, use_cuda, dont_train=False, dont_eval=False, out_path=None, batch_size=batch_size)
            if results["eval_loss"] < best_eval_loss:
                best_eval_loss = results["eval_loss"]
                best_trial = {"lambd": lambd}
            trials[f"contrast_lambd_{lambd}"] = results["eval_loss"]
    else:
        learning_rates = [1e-5 * i for i in range(1, 5)]
        epochs = range(3, 6)
        batch_size = 8

        trials = {}
        best_eval_loss = float("inf")
        best_trial = None

        for lr, num_epochs in zip(learning_rates, epochs):
            #trainer = training_setup(model, tokenizer, model_string, seed, lr, num_epochs, train_path, contrastive_train=contrastive_train, is_hyperparam_opt=False)
            results = main(model_name, "", train_path, eval_path, contrastive_train, 1, num_epochs, seed, lr, use_cuda, dont_train=False, dont_eval=False, out_path=None, batch_size=batch_size)
            if results["eval_loss"] < best_eval_loss:
                best_eval_loss = results["eval_loss"]
                best_trial = {"lr": lr, "num_epochs": num_epochs}
            trials[f"lr_{lr}_epochs_{num_epochs}"] = results["eval_loss"]
        
    return best_trial, trials




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tune the hyperparameters for a model")
    parser.add_argument("model", choices=["gpt2", "gpt-neo-sm", "gpt-neo-lg"]) 
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("-t", "--train_path", default="./lm_train_data/train.txt")
    parser.add_argument("-e", "--eval_path", default="./lm_train_data/dev.txt")

    args = parser.parse_args()
    if args.contrastive:
        best_trial, all_trials = optimize_minimal(args.model, args.cuda, True, args.seed, args.train_path, args.eval_path)
    elif args.model == "gpt2": 
        best_trial = optimize(args.model, args.cuda, args.contrastive, args.seed, args.lr, args.num_epochs, args.train_path, args.eval_path)
        all_trials = {}
    else: # deepspeed + transformers + hyperopt doesn't seem to work together
        #best_trial, all_trials = optimize(args.model, args.cuda, args.contrastive, args.seed, args.lr, args.num_epochs, args.train_path, args.eval_path)
        best_trial = optimize(args.model, args.cuda, args.contrastive, args.seed, args.lr, args.num_epochs, args.train_path, args.eval_path)
        all_trials = {}
        #best_trial, all_trials = optimize_minimal(args.model, args.cuda, args.contrastive, args.seed, args.train_path, args.eval_path)
    records = {"best": best_trial, "all": all_trials}
        
    with open(f"contrast_{args.contrastive}_{args.model}_tuning.p", "wb") as f:
        pickle.dump(records, f)
    print(best_trial)
