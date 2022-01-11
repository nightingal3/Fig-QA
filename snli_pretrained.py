# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
import pdb
import numpy as np


def get_prediction(tokenizer, model, premise, hypothesis, max_length=256):
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        token_type_ids = token_type_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0]  # batch_size only one
    predicted_index = torch.argmax(predicted_probability)
    predicted_probability = predicted_probability.tolist()

    return predicted_probability, predicted_index


if __name__ == '__main__':
    premise = "The girl was as down-to-earth as a Michelin-starred canape."
    hypothesis = "The girl was down to earth."
    metaphor_data = pd.read_csv("./filtered/test.csv")

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    if torch.cuda.is_available():
        model.to('cuda')

    #print(get_prediction(tokenizer, model, premise, hypothesis))
    #assert False
    #snli_dev = []
    #SNLI_DEV_FILE_PATH = "../../data/snli_1.0/snli_1.0_dev.jsonl"   # you can change this to other path.
    #with open(SNLI_DEV_FILE_PATH, mode='r', encoding='utf-8') as in_f:
        #for line in in_f:
            #if line:
                #cur_item = json.loads(line)
                #if cur_item['gold_label'] != '-':
                    #snli_dev.append(cur_item)

    total = 0
    correct = 0
    label_mapping = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction',
    }
    pred1_lst = []
    pred2_lst = []
    choice_preds = []
    entailment_scores = []

    print("Start evaluating...")        # this might take a while.
    for i, item in enumerate(metaphor_data.to_dict(orient="records")):
        print("startphrase: ", item["startphrase"])
        ending1, ending2 = item["ending1"], item["ending2"]

        pred_logits_1, pred_index_end1 = get_prediction(tokenizer, model, item['startphrase'], ending1)
        pred_logits_2, pred_index_end2 = get_prediction(tokenizer, model, item['startphrase'], ending2)
        entailment_score_1 = pred_logits_1[0] - pred_logits_1[2]
        entailment_score_2 = pred_logits_2[0] - pred_logits_2[2]
        choice_pred = 0 if entailment_score_1 > entailment_score_2 else 1

        pred1 = label_mapping[int(pred_index_end1)]
        pred2 = label_mapping[int(pred_index_end2)]

        print(f"ending1: {ending1} is {pred1}")
        print(f"ending2: {ending2} is {pred2}")

        if i % 2 == 0:
            if pred1 == "entailment":
                correct += 1
            if pred2 == "contradiction":
                correct += 1
            entailment_scores.append(entailment_score_1)
        else:
            if pred1 == "contradiction":
                correct += 1
            if pred2 == "entailment":
                correct += 1
            entailment_scores.append(entailment_score_2)
        total += 2
        choice_preds.append(choice_pred)

        pred1_lst.append(pred1)
        pred2_lst.append(pred2)

    print("Total / Correct / Accuracy:", f"{total} / {correct} / {correct / total}")
    print("Correct (binary): ", len(np.where(choice_preds == metaphor_data["labels"])[0])/len(metaphor_data))

    cols = {"startphrase": metaphor_data["startphrase"], "ending1": metaphor_data["ending1"], "ending2": metaphor_data["ending2"],
     "label": metaphor_data["labels"], "ending1_pred": pred1_lst, "ending2_pred": pred2_lst, "choice_pred": choice_preds}
    snli_df = pd.DataFrame(cols)
    snli_df.to_csv("snli_preds.csv", index=False)
