import pandas as pd
import pdb

def write_training_lines_correct(df: pd.DataFrame, out_filename: str, middle_phrase: str = "") -> None:
    second_sentence = [row["ending1"] if row["labels"] == 0 else row["ending2"] for row in df.to_dict(orient="records")]
    format_fn = str.capitalize if middle_phrase == "" else str.lower
    full_sents = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), second_sentence)]
    with open(out_filename, "w") as f:
        f.write("\n".join(full_sents))

def write_training_lines_all(df: pd.DataFrame, out_filename: str, middle_phrase: str = "") -> None:
    # note: important for training file to be in (correct, wrong) order since the model assumes so in contrastive learning
    correct_sentence = [row["ending1"] if row["labels"] == 0 else row["ending2"] for row in df.to_dict(orient="records")]
    wrong_sentence = [row["ending2"] if row["labels"] == 0 else row["ending1"] for row in df.to_dict(orient="records")]
    format_fn = str.capitalize if middle_phrase == "" else str.lower

    sents_correct = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), correct_sentence)]
    sents_wrong = [f"{first_sent.capitalize()}.{middle_phrase}{format_fn(second_sent)}." for first_sent, second_sent in zip(list(df["startphrase"]), wrong_sentence)]

    full_sents = [val for pair in zip(sents_correct, sents_wrong) for val in pair]
    with open(out_filename, "w") as f:
        f.write("\n".join(full_sents))

if __name__ == "__main__":
    df = pd.read_csv("./filtered/mturk_processed - combined.csv")
    valid_data = df.loc[df["valid_all3"] == 1]
    write_training_lines_correct(valid_data, "./lm_train_data/train.txt", "")
    write_training_lines_all(valid_data, "./lm_train_data/train_contrast.txt", "")