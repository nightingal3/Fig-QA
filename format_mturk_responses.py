import pandas as pd
import numpy as np
import pdb

def join_responses(df: pd.DataFrame) -> pd.DataFrame:
    phrase_1_list = [f"{x} {y}".lower() for x, y in zip(df["Answer.metaphor-base"], df["Answer.phrase1"])]
    phrase_2_list = [f"{x} {y}".lower() for x, y in zip(df["Answer.metaphor-base"], df["Answer.phrase2"])]
    ending_1_list = df["Answer.answer1"]
    ending_2_list = df["Answer.answer2"]

    phrases = list(zip(phrase_1_list, phrase_2_list))
    endings = list(zip(ending_1_list, ending_2_list))

    context_col = []
    ending_1_col = []
    ending_2_col = []
    labels = []
    for (phrase_1, phrase_2), (ending_1, ending_2) in zip(phrases, endings):
        context_col.append(phrase_1)
        context_col.append(phrase_2)
        ending_1_col.extend([ending_1] * 2)
        ending_2_col.extend([ending_2] * 2)
        labels.extend([0, 1])

    data = pd.DataFrame({"startphrase": context_col, "ending1": ending_1_col, "ending2": ending_2_col, "labels": labels})
    return data

def join_responses_batch(df: pd.DataFrame, batch_size: int = 5):
    phrase_1_list = [[f"{x} {y}".lower() for x, y in zip(df[f"Answer.metaphor-base-{str(i)}"], df[f"Answer.phrase1-{i}"])] for i in range(1, batch_size + 1)]
    phrase_1_list = sum(phrase_1_list, [])
    phrase_2_list = [[f"{x} {y}".lower() for x, y in zip(df[f"Answer.metaphor-base-{str(i)}"], df[f"Answer.phrase2-{i}"])] for i in range(1, batch_size + 1)]
    phrase_2_list = sum(phrase_2_list, [])
    ending_1_list = [list(df[f"Answer.answer1-{str(i)}"]) for i in range(1, batch_size + 1)]
    ending_1_list = sum(ending_1_list, [])
    ending_2_list = [list(df[f"Answer.answer2-{str(i)}"]) for i in range(1, batch_size + 1)]
    ending_2_list = sum(ending_2_list, [])

    phrases = list(zip(phrase_1_list, phrase_2_list))
    endings = list(zip(ending_1_list, ending_2_list))

    context_col = []
    ending_1_col = []
    ending_2_col = []
    labels = []
    for (phrase_1, phrase_2), (ending_1, ending_2) in zip(phrases, endings):
        context_col.append(phrase_1)
        context_col.append(phrase_2)
        ending_1_col.extend([ending_1] * 2)
        ending_2_col.extend([ending_2] * 2)
        labels.extend([0, 1])

    data = pd.DataFrame({"startphrase": context_col, "ending1": ending_1_col, "ending2": ending_2_col, "labels": labels})
    return data

if __name__ == "__main__":
    df = pd.read_csv("./metaphors_pilot_cut.csv")
    df_5 = pd.read_csv("./mturk_batch_size_5.csv")
    new_df = join_responses(df)
    new_df_5 = join_responses_batch(df_5)
    new_df.to_csv("./train_data_mturk.csv", index=False)
    new_df_5.to_csv("./train_data_mturk_batchsize_5.csv", index=False)