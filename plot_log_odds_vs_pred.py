import pandas as pd
import pdb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

prob_sheets = { #TODO: put GPT-3 here and replace faulty trained sheets
    "gpt2": ["./prob_sheets/gpt2_prob_test.tsv", "./prob_sheets/gpt2_trained_prob_test.csv"],
    "gpt-neo 1.3B": ["./prob_sheets/gpt-neo-sm_prob_test.tsv", "./prob_sheets/gpt-neo-sm_trained_prob_test.csv"],
    #"gpt-3": ["./prob_sheets/gpt3_curie_trained_prob_test.csv", "./prob_sheets/gpt3_curie_trained_prob_test.csv"]
}

def get_log_odds_and_pred(df: pd.DataFrame) -> pd.DataFrame:
    odds = df["P(y_1)"] / df["P(y_2)"]
    log_odds = np.log(odds)
    df["log_odds"] = log_odds
    
    return df[["log_odds", "P(y_1|x_1)", "P(y_2|x_2)"]]

def main(plot_type="trained"):
    log_odds = []
    pred_prob = []
    model_names = []
    for model_name in prob_sheets:
        filepath, filepath_trained = prob_sheets[model_name]
        df_untrained = pd.read_csv(filepath, delimiter="\t")
        df_trained = pd.read_csv(filepath_trained)
        untrained_info = get_log_odds_and_pred(df_untrained)
        trained_info = get_log_odds_and_pred(df_trained)
        info = trained_info if plot_type == "trained" else untrained_info
        #cols[f"{model_name}_odds"] = list(untrained_info["log_odds"])
        log_odds.extend(list(info["log_odds"]))
        pred_prob.extend(list(info["P(y_1|x_1)"]))
        model_names.extend([f"{model_name}_trained"] * len(info))
        
    cols = {"odds": log_odds, "pred": pred_prob, "name": model_names}
    odds_df = pd.DataFrame(cols)
    sns.lmplot(data=odds_df, x="odds", y="pred", hue="name", scatter_kws={'alpha':0.4})
    plt.xlabel("log(P(y_1)/P(y_2))")
    plt.ylabel("P(y_1|x_1)")
    plt.xlim(-2, 3)
    plot_title = "Trained models" if plot_type == "trained" else "Untrained models"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(f"{plot_type}_prob.png")
    plt.savefig(f"{plot_type}_prob.eps")


if __name__ == "__main__":
    main("trained")
    main("untrained")