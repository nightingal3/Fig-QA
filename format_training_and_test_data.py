import pandas as pd

def write_training_lines(df: pd.DataFrame, out_filename: str) -> None:
    second_sentence = [row["ending1"] if row["labels"] == 0 else row["ending2"] for row in df.to_dict(orient="records")]
    full_sents = [f"{first_sent.capitalize()}. {second_sent.capitalize()}." for first_sent, second_sent in zip(list(df["startphrase"]), second_sentence)]
    with open(out_filename, "w") as f:
        f.write("\n".join(full_sents))

if __name__ == "__main__":
    df = pd.read_csv("./filtered/mturk_processed - annotator 1.csv")
    valid_data = df.loc[df["valid"] == 1]
    write_training_lines(valid_data, "./lm_train_data/train.txt")