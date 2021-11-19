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
    model_names = []
    for model_name in prob_sheets:
        filepath, filepath_trained = prob_sheets[model_name]
        df_untrained = pd.read_csv(filepath, delimiter="\t")
        df_trained = pd.read_csv(filepath_trained)
        untrained_info = get_log_odds_and_pred(df_untrained)
        trained_info = get_log_odds_and_pred(df_trained)
        #cols[f"{model_name}_odds"] = list(untrained_info["log_odds"])
        log_odds.extend(list(untrained_info["log_odds"]))
        pred_prob.extend(list(untrained_info["P(y_1|x_1)"]))
        model_names.extend([model_name] * len(untrained_info))
        
    cols = {"odds": log_odds, "pred": pred_prob, "name": model_names}
    odds_df = pd.DataFrame(cols)
    sns.lmplot(data=odds_df, x="odds", y="pred", hue="name")
    plt.xlabel("log(P(y_1)/P(y_2))")
    plt.ylabel("P(y_1|x_1)")
    plt.title("Untrained models")
    plt.xlim(-2, 3)
    plt.tight_layout()
    plt.savefig("untrained.png")

if __name__ == "__main__":
    main()