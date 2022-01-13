import pandas as pd
import pdb

human_error_file = "incorrect_human.csv"
gpt_error_file = "gpt3_incorrect.csv"

if __name__ == "__main__":
    human_err_df = pd.read_csv(human_error_file)
    gpt_err_df = pd.read_csv(gpt_error_file)

    common_errs = human_err_df.merge(gpt_err_df, how="inner", on=["startphrase"])
    human_only_errs = human_err_df.loc[~human_err_df["startphrase"].isin(common_errs["startphrase"])]
    gpt_only_errs = gpt_err_df.loc[~gpt_err_df["startphrase"].isin(common_errs["startphrase"])]

    common_errs.to_csv("common_errors.csv", index=False)
    human_only_errs.to_csv("human_only_errs.csv", index=False)
    gpt_only_errs.to_csv("gpt_only_errs.csv", index=False)
    pdb.set_trace()