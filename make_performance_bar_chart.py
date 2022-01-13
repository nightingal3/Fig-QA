import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

sns.set_theme(style="whitegrid")
sns.set(font_scale = 1.4)
def make_plot(df, out_filepath: str, palette: str = "dark", plot_type: str = "finetune") -> None: # from this example: https://seaborn.pydata.org/examples/grouped_barplot.html
    if plot_type == "finetune":
        order = ["GPT-2", "GPT-neo 1.3B", "GPT-3 Ada", "GPT-3 Babbage", "GPT-3 Curie", "GPT-3 Davinci", "BERT", "RoBERTa", "Human", "Human (confident)"]
    elif plot_type == "prompt":
        order = ["GPT-2", "GPT-neo 1.3B", "GPT-3 Ada"]
    elif plot_type == "transfer":
        order = ["BERT", "RoBERTa"]

    g = sns.catplot(
    data=df, kind="bar",
    x="Model", y="Accuracy", hue="Method",
    ci="sd", palette=palette, height=6, order=order, col_order=["Zero-shot", "Transfer", "Fine-tuning", "Human"]
    )
    g.despine(left=True)
    g.set_axis_labels("", "Accuracy")
    g._legend.remove()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.ylim(0.5, 1)
    if plot_type != "prompt":
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{out_filepath}.png")
    plt.savefig(f"{out_filepath}.eps")

if __name__ == "__main__":
    df = pd.read_csv("./metaphor_performance_results.csv")
    df_zero_shot_finetune = df.loc[((df["Method"] == 'Zero-shot') | (df["Method"] == "Fine-tuning")) | ((df["Method"] == "Human") | (df["Method"].str.contains("Transfer")))]
    df_prompting = df[(df["Method"].str.contains("Prompt")) | (df["Method"] == "Zero-shot")]
    df_transfer = df[df["Method"].str.contains("Transfer")]

    make_plot(df_zero_shot_finetune, "performance_barchart_zeroshot_finetune")
    make_plot(df_prompting, "prompting_bar", "rocket", plot_type = "prompt")
    #make_plot(df_transfer, "transfer_bar", "crest", plot_type = "transfer")