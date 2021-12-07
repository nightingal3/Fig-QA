import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import seaborn as sns
import pdb

def format_prob_data(prob_sheet_list: List, direction: str = "forward") -> pd.DataFrame:
    model_types = []
    conditions = []
    confidence = []
    correctness = []
    for sheet_path in prob_sheet_list:
        if "gpt2" in sheet_path:
            model_type = "gpt2"
        elif "gpt-neo" in sheet_path:
            model_type = "gpt-neo"
        else:
            model_type = "gpt3"
        
        if "trained" in sheet_path:
            condition = "trained"
        else:
            condition = "untrained"

        sep = "\t" if ".tsv" in sheet_path else ","
        df = pd.read_csv(sheet_path, sep=sep)
        correct_col_1 = "P(y_1|x_1)" if direction == "forward" else "P(x_1|y_1)"
        correct_col_2 = "P(y_2|x_2)" if direction == "forward" else "P(x_2|y_2)"

        df_1_correct = df[correct_col_1].apply(lambda x: x > 0.5).replace({True: "Yes", False: "No"})
        confidence.extend(list(df[correct_col_1]))
        correctness.extend(list(df_1_correct))
        model_types.extend([model_type] * len(df))
        conditions.extend([condition] * len(df))

        df_2_correct = df[correct_col_2].apply(lambda x: x > 0.5).replace({True: "Yes", False: "No"})
        confidence.extend(list(df[correct_col_2]))
        correctness.extend(list(df_2_correct))
        model_types.extend([model_type] * len(df))
        conditions.extend([condition] * len(df))

    return pd.DataFrame({"model_type": model_types, "condition": conditions, "P(correct|context)": confidence, "correct": correctness})

def plot_violin(data: pd.DataFrame, out_filename: str) -> None:
    plt.gcf().clear()
    g = sns.catplot(x="model_type", y="P(correct|context)",
                hue="correct", col="condition",
                data=data, kind="violin", split=True,
                height=4, aspect=.7, palette="Pastel1", scale="count")
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")


if __name__ == "__main__":
    prob_sheets = ["./prob_sheets/gpt2_prob_test.tsv", "./prob_sheets/gpt2_trained_prob_test.csv",
    "./prob_sheets/gpt-neo-sm_prob_test.tsv","./prob_sheets/gpt-neo-sm_trained_prob_test.csv", "./prob_sheets/gpt3_curie_trained_prob_test.csv"]
    prob_df = format_prob_data(prob_sheets, direction="forward")
    prob_df_back = format_prob_data(prob_sheets, direction="backward")
    plot_violin(prob_df, "forward_violin")
    plot_violin(prob_df_back, "backward_violin")