import pandas as pd
import os
import pdb
import json

prob_sheets_path = "./data-copy/prob_sheets"
#commonsense_path = "./data/commonsense_annotation/commonsense_test.csv"

def binary_preds_to_system_output(in_filename: str, out_filename: str, reference_sheet: str) -> None:
    all_qs = []
    reference_df = pd.read_csv(reference_sheet)
    with open(in_filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            pred = int(line.strip())
            bin_pred = 0 if pred == 1 else 1
            question_text = reference_df.iloc[i]["startphrase"]
            label = reference_df.iloc[i]["labels"]
            answer = reference_df.iloc[i]["ending1"] if label == 0 else reference_df.iloc[i]["ending2"]
            pred_text = reference_df.iloc[i]["ending1"] if bin_pred == 0 else reference_df.iloc[i]["ending2"]

            q = {
            "context": question_text,
            "question": "",
            "answers": {
                "text": answer,
                "option_index": int(label)
            },
            "options": [
                reference_df.iloc[i]["ending1"],
                reference_df.iloc[i]["ending2"]
            ],
            "predicted_answers": {
                "text": pred_text,
                "option_index": int(bin_pred)
            },
            #"data_features": {
                #"commonsense_category": commonsense_cat_q1
            #}
            }
            all_qs.append(q)

    json_string = json.dumps(all_qs, indent=4)
    with open(out_filename, "w") as out_f:
        out_f.write(json_string)


def prob_sheet_to_system_output(in_filename: str, out_filename: str, commonsense_filename: str) -> None:
    sep = "\t" if in_filename.endswith("tsv") else ","
    df = pd.read_csv(in_filename, sep=sep)
    if commonsense_filename is not None:
        commonsense_df = pd.read_csv(commonsense_filename)
    all_qs = []
    for row in df.to_dict(orient="records"):
        x1, x2 = row["x_1"], row["x_2"]
        y1, y2 = row["y_1"], row["y_2"]
        if commonsense_filename is not None:
            commonsense_cat_q1 = commonsense_df.loc[commonsense_df["startphrase"] == row["x_1"]][["obj", "vis", "soc", "cul"]]
            commonsense_cat_q2 = commonsense_df.loc[commonsense_df["startphrase"] == row["x_2"]][["obj", "vis", "soc", "cul"]]
            commonsense_cat_q1 = list(commonsense_cat_q1.dropna(axis=1).columns)
            commonsense_cat_q2 = list(commonsense_cat_q2.dropna(axis=1).columns)

        pred_x1_text = y1 if row["P(y_1|x_1)"] > 0.5 else y2
        pred_x1_option = 0 if row["P(y_1|x_1)"] > 0.5 else 1 

        pred_x2_text = y2 if row["P(y_2|x_2)"] > 0.5 else y1
        pred_x2_option = 1 if row["P(y_2|x_2)"] > 0.5 else 0 

        q1 = {
            "context": x1,
            "question": "",
            "answers": {
                "text": y1,
                "option_index": 0
            },
            "options": [
                y1,
                y2
            ],
            "predicted_answers": {
                "text": pred_x1_text,
                "option_index": pred_x1_option
            },
            #"data_features": {
                #"commonsense_category": commonsense_cat_q1
            #}
        }
        all_qs.append(q1)

        q2 = {
            "context": x2,
            "question": "",
            "answers": {
                "text": y2,
                "option_index": 1
            },
            "options": [
                y1,
                y2
            ],
            "predicted_answers": {
                "text": pred_x2_text,
                "option_index": pred_x2_option
            },
            #"data_features": {
                #"commonsense_category": commonsense_cat_q2
            #}
        }
        all_qs.append(q2)
    
    json_string = json.dumps(all_qs, indent=4)
    with open(out_filename, "w") as out_f:
        out_f.write(json_string)


if __name__ == "__main__":
    binary_preds_to_system_output("./data-copy/dev_set/models_metaphor_m_bert_lr2e-5_0/predictions_dev.lst", "./data-copy/system_outputs/bert_dev.json", "./data-copy/filtered/dev.csv")
    binary_preds_to_system_output("./data-copy/dev_set/models_metaphor_m_roberta_lr5e-6_0/predictions_dev.lst", "./data-copy/system_outputs/roberta_dev.json", "./data-copy/filtered/dev.csv")

    out_dir = "./data-copy/system_outputs"
    for filename in os.listdir(prob_sheets_path):
        if "dev" not in filename:
            continue
        out_name = f"{filename[:-4]}.json"
        prob_sheet_to_system_output(f"{prob_sheets_path}/{filename}", f"{out_dir}/{out_name}", None)