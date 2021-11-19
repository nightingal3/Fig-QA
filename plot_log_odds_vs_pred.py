import pandas as pd
import pdb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

prob_sheets = {
    "gpt2": ["./prob_sheets/gpt2_prob.tsv", "./prob_sheets/gpt_2_trained.csv"],
    "gpt-neo 1.3B": ["./prob_sheets/neo-sm_prob.tsv", "./prob_sheets/neo_sm_trained.csv"],
    "gpt-neo 2.7B": ["./prob_sheets/neo-lg_prob.tsv", "./prob_sheets/neo_lg_trained.csv"],
}

def get_log_odds_and_pred(df: pd.DataFrame) -> pd.DataFrame:
    odds = df["P(y_1)"] / df["P(y_2)"]
    log_odds = np.log(odds)
    df["log_odds"] = log_odds
    
    return df[["log_odds", "P(y_1|x_1)", "P(y_2|x_2)"]]

def main():
    log_odds = []
    pred_prob = []
    model_name = []
    for model_name in prob_sheets:
        filepath, filepath_trained = prob_sheets[model_name]
        df_untrained = pd.read_csv(filepath, delimiter="\t")
        df_trained = pd.read_csv(filepath_trained)
        untrained_info = get_log_odds_and_pred(df_untrained)
        trained_info = get_log_odds_and_pred(df_trained)
        #cols[f"{model_name}_odds"] = list(untrained_info["log_odds"])
        log_odds.extend(list(trained_info["log_odds"]))
        pred_prob.extend(list(trained_info["P(y_1|x_1)"]))
        model_name.extend([f"{model_name}_trained"] * len(trained_info))
        
    cols = {"odds": log_odds, "pred": pred_prob, "name": model_name}
    odds_df = pd.DataFrame(cols)
    pdb.set_trace()
    sns.lineplot(data=odds_df, x="odds", y="pred", hue="name")
    plt.xlabel("log(P(y_1)/P(y_2))")
    plt.ylabel("P(y_1|x_1)")
    plt.xlim(-2, 3)
    plt.title("Trained models")

if __name__ == "__main__":
    main()