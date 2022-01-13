import pickle
import pdb
import matplotlib.pyplot as plt
from typing import List

def plot_loss_curve(loss_data: List, out_filename: str) -> None:
    eval_loss_standard = []
    epochs_standard = []

    for x in loss_data:
        if "eval_loss" in x:
            eval_loss_standard.append(x["eval_loss"])
            epochs_standard.append(x["epoch"])

    plt.plot(epochs_standard, eval_loss_standard)
    plt.xlabel("Epochs")
    plt.ylabel("Eval loss")
    plt.savefig(out_filename)

if __name__ == "__main__":
    gpt_2_standard_lr = pickle.load(open("./gpt2_epochs_8_eval_loss.p", "rb"))
    gpt_2_2e_5 = pickle.load(open("./gpt2_epochs_10_eval_loss.p", "rb"))
    gpt_neo_standard = pickle.load(open("./gpt-neo-sm_epochs_8_eval_loss.p", "rb"))
    
    plot_loss_curve(gpt_neo_standard, "gpt-neo_loss.png")