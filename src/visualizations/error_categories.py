import pandas as pd
import pdb
import os

def count_errors(error_df: pd.DataFrame, category_df: pd.DataFrame) -> dict:
    error_df = error_df.merge(category_df, on=["startphrase"], how="left")[["obj", "vis", "soc", "cul"]]
    error_df = error_df.fillna(0)
    total_num = len(error_df)

    obj_errors = error_df["obj"].sum()/total_num
    vis_errors = error_df["vis"].sum()/total_num
    soc_errors = error_df["soc"].sum()/total_num
    cul_errors = error_df["cul"].sum()/total_num

    return {"obj": obj_errors, "vis": vis_errors, "soc": soc_errors, "cul": cul_errors}

def count_errors_by_category(error_df: pd.DataFrame, category_df: pd.DataFrame) -> dict:
    error_df = error_df.merge(category_df, on=["startphrase"], how="left")[["obj", "vis", "soc", "cul"]]
    error_df = error_df.fillna(0)

    total_obj = category_df["obj"].sum()
    total_vis = category_df["vis"].sum()
    total_soc = category_df["soc"].sum()
    total_cul = category_df["cul"].sum()

    obj_acc = 1 - (error_df["obj"].sum()/total_obj)
    vis_acc = 1 - (error_df["vis"].sum()/total_vis)
    soc_acc = 1 - (error_df["soc"].sum()/total_soc)
    cul_acc = 1 - (error_df["cul"].sum()/total_cul)

    return {"obj": obj_acc, "vis": vis_acc, "soc": soc_acc, "cul": cul_acc}

def gen_error_file_from_prob(prob_df: pd.DataFrame) -> pd.DataFrame:
    startphrase_lst = []
    ending1_lst = []
    ending2_lst = []
    correct_labels = []
    for row in prob_df.to_dict(orient="records"):
        if row["P(y_1|x_1)"] <= 0.5: # first prompt is wrong
            startphrase_lst.append(row["x_1"])
            correct_labels.append(0)
            ending1_lst.append(row["y_1"])
            ending2_lst.append(row["y_2"])
        if row["P(y_2|x_2)"] <= 0.5: # second prompt is wrong
            startphrase_lst.append(row["x_2"])
            correct_labels.append(1)
            ending1_lst.append(row["y_1"])
            ending2_lst.append(row["y_2"])
    return pd.DataFrame({"startphrase": startphrase_lst, "ending1": ending1_lst, "ending2": ending2_lst, "correct_label": correct_labels})

def gen_all_error_files(in_dir: str, out_dir: str) -> None:
    for f in os.listdir(in_dir):
        if f.endswith(".csv"):
            sep = ","
        elif f.endswith(".tsv"):
            sep = "\t"
        else:
            continue
        out_name = f[:-4]
        out_filename = f"{out_dir}/{out_name}.csv"
        prob_df = pd.read_csv(f"{in_dir}/{f}", sep=sep)
        error_df = gen_error_file_from_prob(prob_df)
        error_df.to_csv(out_filename, index=False)


if __name__ == "__main__":
    human_errors = pd.read_csv("./commonsense_annotation/incorrect_human.csv")

    gpt3_errors = pd.read_csv("./commonsense_annotation/gpt3_curie_incorrect.csv")
    gpt3_errors_trained = pd.read_csv("./commonsense_annotation/gpt3_curie_trained_incorrect.csv")

    gpt2_errors = pd.read_csv("./commonsense_annotation/gpt2_incorrect.csv")
    gpt2_errors_trained = pd.read_csv("./commonsense_annotation/gpt2_trained_incorrect.csv")

    gptneo_errors = pd.read_csv("./commonsense_annotation/gpt_neo_sm_incorrect.csv")
    gptneo_errors_trained = pd.read_csv("./commonsense_annotation/gpt_neo_sm_trained_incorrect.csv")

    roberta_errors = pd.read_csv("./commonsense_annotation/roberta_incorrect.csv")
    roberta_errors = roberta_errors.loc[roberta_errors["pred is correct"] == 0]

    bert_errors = pd.read_csv("./commonsense_annotation/bert_incorrect.csv")
    bert_errors = bert_errors.loc[bert_errors["bert correct"] == 0]

    commonsense_cat = pd.read_csv("./commonsense_annotation/commonsense_test.csv")

    error_cat_human = count_errors_by_category(human_errors, commonsense_cat)
    error_cat_gpt3 = count_errors_by_category(gpt3_errors, commonsense_cat)
    error_cat_gpt3_trained = count_errors_by_category(gpt3_errors_trained, commonsense_cat)
    error_cat_gpt2 = count_errors_by_category(gpt2_errors, commonsense_cat)
    error_cat_gpt2_trained = count_errors_by_category(gpt2_errors_trained, commonsense_cat)
    error_cat_gptneo = count_errors_by_category(gptneo_errors, commonsense_cat)
    error_cat_gptneo_trained = count_errors_by_category(gptneo_errors_trained, commonsense_cat)
    error_cat_roberta = count_errors_by_category(roberta_errors, commonsense_cat)
    error_cat_bert = count_errors_by_category(bert_errors, commonsense_cat)

    print("UNTRAINED")
    print("human: ", error_cat_human)
    print("gpt2: ", error_cat_gpt2)
    print("gptneo: ", error_cat_gptneo)
    print("gpt3: ", error_cat_gpt3)
    print("TRAINED")
    print("gpt2: ", error_cat_gpt2_trained)
    print("gptneo: ", error_cat_gptneo_trained)    
    print("gpt3: ", error_cat_gpt3_trained)
    print("roberta: ", error_cat_roberta)
    print("bert: ", error_cat_bert)
