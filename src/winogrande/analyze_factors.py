import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('averaged_perceptron_tagger')
import pdb

def select_random_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n)

def acc_by_length(df: pd.DataFrame) -> None:
    df["len"] = df["sentence"].str.count(" ") + 1
    quantiles = [df["len"].quantile(x * 0.2) for x in range(1, 6)]

    acc = []
    for i, quantile in enumerate(quantiles[:-1]):
        selected = df.query(f"len >= {quantile} & len < {quantiles[i + 1]}")
        acc_quantile = len(selected.loc[selected["answer==pred"] == "TRUE"])/len(selected)
        acc.append(acc_quantile)

    plt.bar([0.2, 0.4, 0.6, 0.8], acc, width=0.1)
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.xlabel("Quantile of sentence length (words)")
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.savefig("length.png")

def acc_by_answer_pos(df: pd.DataFrame) -> None:
    df["pos1"] = df["option1"].apply(get_pos)
    df["pos2"] = df["option2"].apply(get_pos)
    df["combinedPOS"] = df["pos1"] + "_" + df["pos2"]

    answer_nouns = df.query("pos1 == 'NN' & pos2 == 'NN'")
    acc_nouns = len(answer_nouns.loc[answer_nouns["answer==pred"] == "TRUE"])/len(answer_nouns)
    answer_proper_nouns = df.query("pos1 == 'NNP' & pos2 == 'NNP'")
    acc_proper_nouns = len(answer_proper_nouns.loc[answer_proper_nouns["answer==pred"] == "TRUE"])/len(answer_proper_nouns)
    
    plt.bar([0, 1], [acc_nouns, acc_proper_nouns])
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1)
    plt.xticks([0, 1], labels=["Nouns", "Proper Nouns"])
    plt.savefig("pos.png")

def frequency_alignment(df: pd.DataFrame) -> None:
    assert NotImplementedError

def get_pos(text: str) -> str:
    if isinstance(text, str):
        #return nltk.pos_tag([text])[0][1]
        return "NNP" if text[0].isupper() else "NN"
    return "none"

if __name__ == "__main__":
    df = pd.read_csv("dev_preds.csv")
    #df_new = pd.read_csv("dev_preds_partially_labelled.csv")
    #sample = select_random_subset(df_new, 100)
    #sample.to_csv("random_sample.csv", index=False)
    #assert False
    acc_by_length(df)
    plt.gcf().clear()
    acc_by_answer_pos(df)