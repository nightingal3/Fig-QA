# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
import pdb
from tqdm import tqdm


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
    preds = []

    print("Start evaluating...")        # this might take a while.
    for i, item in tqdm(enumerate(metaphor_data.to_dict(orient="records"))):
        print("startphrase: ", item["startphrase"])
        ending1, ending2 = item["ending1"], item["ending2"]

        _, pred_index_end1 = get_prediction(tokenizer, model, item['startphrase'], ending1)
        _, pred_index_end2 = get_prediction(tokenizer, model, item['startphrase'], ending2)
        pred1 = label_mapping[int(pred_index_end1)]
        pred2 = label_mapping[int(pred_index_end2)]

        print(f"ending1: {ending1} is {pred1}")
        print(f"ending2: {ending2} is {pred2}")

        if i % 2 == 0:
            if pred1 == 0:
                correct += 1
            if pred2 == 1:
                correct += 1
        else:
            if pred1 == 1:
                correct += 1
            if pred2 == 0:
                correct += 1
        total += 2

        preds.extend([pred1, pred2])

    print("Total / Correct / Accuracy:", f"{total} / {correct} / {correct / total}")
