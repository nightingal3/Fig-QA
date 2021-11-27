import pandas as pd
import pdb
from sklearn.model_selection import train_test_split

def write_training_lines_correct(df: pd.DataFrame, out_filename: str, middle_phrase: str = "") -> None:
    second_sentence = [row["ending1"].replace(".", "") if row["labels"] == 0 else row["ending2"].replace(".", "") for row in df.to_dict(orient="records")]
    df["startphrase"] = df["startphrase"].str.rstrip(",.! \n\t")
    format_fn = str.capitalize if middle_phrase == "" else str.lower
    full_sents = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), second_sentence)]
    with open(out_filename, "w") as f:
        f.write("\n".join(full_sents))

def write_training_lines_all(df: pd.DataFrame, out_filename: str, middle_phrase: str = "") -> None:
    # note: important for training file to be in (correct, wrong) order since the model assumes so in contrastive learning
    correct_sentence = [row["ending1"].replace(".", "") if row["labels"] == 0 else row["ending2"].replace(".", "") for row in df.to_dict(orient="records")]
    wrong_sentence = [row["ending2"].replace(".", "") if row["labels"] == 0 else row["ending1"].replace(".", "") for row in df.to_dict(orient="records")]
    format_fn = str.capitalize if middle_phrase == "" else str.lower
    df["startphrase"] = df["startphrase"].str.rstrip(",.! \n\t")
    sents_correct = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), correct_sentence)]
    sents_wrong = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), wrong_sentence)]

    full_sents = [val for pair in zip(sents_correct, sents_wrong) for val in pair]
    with open(out_filename, "w") as f:
        f.write("\n".join(full_sents))

def format_with_prompts(df: pd.DataFrame, prefix_prompt: str, middle_prompt: str, out_filename: str, correct_only: bool = False) -> None:
    startphrases = prefix_prompt + df["startphrase"] 
    df["startphrase"] = startphrases
    if correct_only:
        write_training_lines_correct(df, out_filename, middle_prompt)
    else:
        write_training_lines_all(df, out_filename, middle_prompt)

def split_df(df, test_and_eval_size=0.6) -> None:
    qids = list(df["qid"].unique())
    train_inds, test_inds = train_test_split(qids, test_size=test_and_eval_size)
    dev_inds, test_inds = train_test_split(test_inds, test_size=0.5)

    train = df.loc[df["qid"].isin(train_inds)]
    test = df.loc[df["qid"].isin(test_inds)]
    dev = df.loc[df["qid"].isin(dev_inds)]

    train.to_csv("./filtered/train.csv", index=False)
    test.to_csv("./filtered/test.csv", index=False)
    dev.to_csv("./filtered/dev.csv", index=False)

    return train, dev, test

if __name__ == "__main__":
    df = pd.read_csv("./filtered/annotators_combined.csv")
    valid_data = df.loc[df["valid"] == 1]
    pilot_examples = pd.read_csv("./filtered/original_data.csv", sep="\t")
    valid_data.index = range(len(valid_data))
    valid_data["qid"] = valid_data.index // 2
    pilot_examples["qid"] = pilot_examples.index // 2
    pilot_examples["qid"] = "P" + pilot_examples["qid"].astype(str)

    train, dev, test = split_df(valid_data)
    test = pd.concat([test, pilot_examples])
    test.to_csv("./filtered/test.csv", index=False)

    write_training_lines_correct(train, "./lm_train_data/train.txt", "")
    write_training_lines_all(train, "./lm_train_data/train_contrast.txt", "")
    write_training_lines_correct(dev, "./lm_train_data/dev.txt", "")
    write_training_lines_correct(test, "./lm_train_data/test.txt", "")
