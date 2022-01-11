import pandas as pd
from scipy.stats import pointbiserialr
import pdb

if __name__ == "__main__":
    df = pd.read_csv("./prob_sheets/gpt-neo-sm_trained_prob_test.csv", sep=",")
    df["len_startphrase_1"] = df["x_1"].str.split().str.len()
    df["len_startphrase_2"] = df["x_2"].str.split().str.len()

    startphrase_len = list(df["len_startphrase_1"]) + list(df["len_startphrase_2"])
    correct_p = list(df["P(y_1|x_1)"]) + list(df["P(y_2|x_2)"]) 
    print("len correlation: ", pointbiserialr(startphrase_len, correct_p))

    answer_prob = list(df["P(y_1)"]) + list(df["P(y_2)"])
    print("answer prob correlation: ", pointbiserialr(answer_prob, correct_p))

    segmented_file = "./dataset_characteristics/train_set_segmented.csv"
    df_segment = pd.read_csv(segmented_file)
    #pdb.set_trace()
    #merged = pd.merge(df, df_segment, how="left", right_on="ending1", left_on="x_1")
    #pdb.set_trace()