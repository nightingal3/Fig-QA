import pandas as pd
import pdb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

prob_sheets = { #TODO: put GPT-3 here and replace faulty trained sheets
    "gpt2": ["./prob_sheets/gpt2_prob_test.tsv", "./prob_sheets/gpt2_trained_prob_test.csv"],
    "gpt-neo 1.3B": ["./prob_sheets/gpt-neo-sm_prob_test.tsv", "./prob_sheets/gpt-neo-sm_trained_prob_test.csv"],
    "gpt3": ["./prob_sheets/gpt3_probabilities_curie_test.csv", "./prob_sheets/gpt3_probabilities_finedtuned_curie_test.csv"]
}

sns.set(font_scale=1.9)

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
        if model_name != "gpt3":
            df_untrained = pd.read_csv(filepath, delimiter="\t")
        else:
            df_untrained = pd.read_csv(filepath, delimiter=",")
        df_trained = pd.read_csv(filepath_trained)

        untrained_info = get_log_odds_and_pred(df_untrained)
        trained_info = get_log_odds_and_pred(df_trained)
        info = trained_info if plot_type == "trained" else untrained_info
        model_id = f"{model_name}_trained" if plot_type == "trained" else model_name

        log_odds.extend(list(info["log_odds"]))
        pred_prob.extend(list(info["P(y_1|x_1)"]))
        model_names.extend([model_id] * len(info))


    cols = {"odds": log_odds, "pred": pred_prob, "name": model_names}
    odds_df = pd.DataFrame(cols)
    gpt_2_df = odds_df.loc[(odds_df["name"] == "gpt2") | (odds_df["name"] == "gpt2_trained")]
    print(spearmanr(gpt_2_df["odds"], gpt_2_df["pred"]))

    gpt_neo_df = odds_df.loc[(odds_df["name"] == "gpt-neo 1.3B") | (odds_df["name"] == "gpt-neo 1.3B_trained")]
    print(spearmanr(gpt_neo_df["odds"], gpt_neo_df["pred"]))

    gpt_3_df = odds_df.loc[(odds_df["name"] == "gpt3") | (odds_df["name"] == "gpt3_trained")]
    print(spearmanr(gpt_3_df["odds"], gpt_3_df["pred"]))

    g = sns.lmplot(data=odds_df, x="odds", y="pred", hue="name", row="name")
    g.axes[0,0].set_xlabel("log(P(y_1)/P(y_2))")
    g.axes[0,0].set_ylabel("P(y_1|x_1)")
    g.axes[1,0].set_xlabel("log(P(y_1)/P(y_2))")
    g.axes[1,0].set_ylabel("P(y_1|x_1)")
    g.axes[2,0].set_xlabel("log(P(y_1)/P(y_2))")
    g.axes[2,0].set_ylabel("P(y_1|x_1)")
    plt.xlim(-2, 3)
    #g._legend.remove()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
        #fancybox=True, shadow=True)
    plot_title = "Trained models" if plot_type == "trained" else "Untrained models"
    #g.title(plot_title)
    plt.tight_layout()
    plt.savefig(f"{plot_type}_prob.png")
    plt.savefig(f"{plot_type}_prob.eps")


if __name__ == "__main__":
    main("trained")
    main("untrained")