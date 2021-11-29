import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

def make_plot(df, out_filepath: str) -> None: # from this example: https://seaborn.pydata.org/examples/grouped_barplot.html
    g = sns.catplot(
    data=df, kind="bar",
    x="Model", y="Accuracy", hue="Method",
    ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("Method", "Accuracy")
    g.legend.set_title("")
    plt.savefig(f"{out_filepath}.png")
    plt.savefig(f"{out_filepath}.eps")

if __name__ == "__main__":
    df = pd.read_csv("./metaphor_performance_results.csv")
    make_plot(df, "./performance_barchart")