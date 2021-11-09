import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPTNeoForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from scipy.special import softmax
from sample_metaphors import trial_dataset
import pdb
import pandas as pd
import pkg_resources
from transformers import file_utils
import math
from typing import List

# current code is from this gist! https://gist.github.com/yuchenlin/eb63e2d0513f70cfc9bb85fa5a78953b
# need to modify for the specific use case

def model_init(model_string, cuda, output_attentions=False, fast=False):
    if model_string.startswith("gpt2"):
        if fast:
            tokenizer = AutoTokenizer.from_pretrained(model_string)
            model = GPT2LMHeadModel.from_pretrained(model_string)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_string) 
            model = GPT2LMHeadModel.from_pretrained(model_string)
    elif model_string.startswith("EleutherAI/gpt-neo"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string, output_attentions=output_attentions)
        #model = AutoModelForCausalLM.from_pretrained(model_string)
        model = GPTNeoForCausalLM.from_pretrained(model_string, output_attentions=output_attentions)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    model.eval()
    if cuda:
        model.to('cuda')
    return model, tokenizer


def sent_scoring(model_tokenizer, text, cuda, score_type="loss", output_attentions=False):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    assert model is not None
    assert tokenizer is not None
    encoded_text = tokenizer.encode(text)
    input_ids = torch.tensor(encoded_text).unsqueeze(0) 
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_attentions=output_attentions)
    loss, logits = outputs[:2]

    sentence_prob = loss.item()
    if score_type == "prob":
        sentence_prob = math.exp(-1.0 * loss)

    if output_attentions:
        attn = outputs["attentions"]
        return sentence_prob, attn, input_ids
        
    return sentence_prob

def confusion_matrix(P_forward_1, P_forward_2, P_backward_1, P_backward_2):
    correct_forward = len(np.where(np.array(P_forward_1) >= 0.5)[0]) + len(np.where(np.array(P_forward_2) >=0.5)[0])
    wrong_forward = len(P_forward_1) + len(P_forward_2) - correct_forward

    correct_backward = len(np.where(np.array(P_backward_1) >= 0.5)[0]) + len(np.where(np.array(P_backward_2) >=0.5)[0])
    wrong_backward = len(P_backward_1) + len(P_backward_2) - correct_backward

    print("correct forward", correct_forward, "wrong forward", wrong_forward, "correct backward", correct_backward, "wrong_backward", wrong_backward)

def evaluate_model(model, tokenizer, test_set, middle_phrase="", verbose=True, score_type="prob", use_cuda=False, acc_only=False) -> tuple:
    preds = []
    labels = []
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    P_x_1 = []
    P_x_2 = []
    P_y_1 = []
    P_y_2 = []
    P_x_1_y_1 = []
    P_x_1_y_2 = []
    P_x_2_y_1 = []
    P_x_2_y_2 = []
    P_x_1_correct = []
    P_x_2_correct = []
    P_y_1_correct = []
    P_y_2_correct = []
    correct = 0

    for i, metaphor_data in enumerate(test_set):
        ctx, p1, p2 = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"]
        labels.append(int(metaphor_data["label"]))
        
        sent1 = ctx + ". " + middle_phrase + p1 + "."
        sent2 = ctx + ". " + middle_phrase + p2 + "."

        score1 = sent_scoring((model, tokenizer), sent1, use_cuda, score_type=score_type)
        score2 = sent_scoring((model, tokenizer), sent2, use_cuda, score_type=score_type)

        if score_type == "loss":
            pred = 0 if score1 < score2 else 1
        else:
            pred = 1 if score1 < score2 else 0

        pred_sent = sent1 if pred == 0 else sent2

        if i % 2 == 0:
            x_1.append(ctx)
            x_1_score = sent_scoring((model, tokenizer), ctx + ".", use_cuda, score_type=score_type)
            P_x_1.append(x_1_score)
            y_1.append(p1)
            y_2.append(p2)
            y1_score = sent_scoring((model, tokenizer), p1 + ".", use_cuda, score_type=score_type)
            y2_score = sent_scoring((model, tokenizer), p2 + ".", use_cuda, score_type=score_type)
            P_y_1.append(y1_score)
            P_y_2.append(y2_score)

            P_x_1_y_1.append(score1)
            P_x_1_y_2.append(score2)
            P_x_1_correct.append(score1/(score1 + score2))

        else:
            x_2.append(ctx)
            x_2_score = sent_scoring((model, tokenizer), ctx + ".", use_cuda, score_type=score_type)
            P_x_2.append(x_2_score)
            P_x_2_y_1.append(score1)
            P_x_2_y_2.append(score2)
            P_x_2_correct.append(score2/(score1 + score2))

            P_y_1_correct.append(P_x_1_y_1[-1]/(P_x_1_y_1[-1] + score1))
            P_y_2_correct.append(score2/(P_x_1_y_2[-1] + score2))
        
        if verbose:
            print(f"Q: {ctx}: 1. {p1} 2. {p2}")
            print(f"model says '{pred_sent}' is more likely")
            print("\n")
        if pred == metaphor_data["label"]:
            correct += 1
        preds.append(pred)

    cols = {"x_1": x_1, "x_2": x_2, "y_1": y_1, "y_2": y_2, "P(x_1)": P_x_1, "P(x_2)": P_x_2, "P(y_1)": P_y_1, "P(y_2)": P_y_2,
        "P(x_1, y_1)": P_x_1_y_1, "P(x_1, y_2)": P_x_1_y_2, "P(x_2, y_1)": P_x_2_y_1, "P(x_2, y_2)": P_x_2_y_2,
        "P(y_1|x_1)": P_x_1_correct, "P(y_2|x_2)": P_x_2_correct, "P(x_1|y_1)": P_y_1_correct, "P(x_2|y_2)": P_y_2_correct}
    out_df = pd.DataFrame(cols)

    if acc_only:
        return correct/len(labels)

    return out_df, preds, labels
 
def compute_stats(total_df: pd.DataFrame, all_preds: List, all_labels: List) -> None:
    print("overall accuracy: ")
    print(len(np.where(np.array(all_preds) == np.array(all_labels))[0])/len(all_labels))
    print("confusion matrix: ")
    confusion_matrix(list(total_df["P(y_1|x_1)"]), list(total_df["P(y_2|x_2)"]), list(total_df["P(x_1|y_1)"]), list(total_df["P(x_2|y_2)"]))

if __name__ == '__evaluate_model__':
    # model, tokenizer = model_init('openai-gpt', False) 
    #model, tokenizer = model_init('gpt2', False) 
    use_cuda = False
    model_names = {"gpt2": "gpt2", "neo-sm": "EleutherAI/gpt-neo-1.3B", "neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_id = "neo-lg"
    model_name = model_names[model_id]
    tsv_name = f"{model_id}_prob"
    middle_phrase = "That is to say, "
    score_type = "prob"
    verbose=False

    model, tokenizer = model_init(model_name, use_cuda)
    metaphor_set = trial_dataset["test"]
    mturk_data = pd.read_csv("train_data_mturk.csv")
    mturk_data_2 = pd.read_csv("train_data_mturk_batchsize_5.csv")
    # the second batch had a lot of nonsensical ones, filtering them out
    mturk_data_2 = mturk_data_2.loc[mturk_data_2["valid"] == 1] 
    
    # without interjection
    my_examples_df, preds_1, labels_1 = evaluate_model(model, tokenizer, metaphor_set, use_cuda=use_cuda, verbose=verbose)
    mturk_examples_df, preds_2, labels_2 = evaluate_model(model, tokenizer, mturk_data.to_dict(orient="records"), use_cuda=use_cuda, verbose=verbose)
    mturk_examples_df_2, preds_3, labels_3 = evaluate_model(model, tokenizer, mturk_data_2.to_dict(orient="records"), use_cuda=use_cuda, verbose=verbose)
    
    all_preds = sum([preds_1, preds_2, preds_3], [])
    all_labels = sum([labels_1, labels_2, labels_3], [])
    total_df = pd.concat([my_examples_df, mturk_examples_df, mturk_examples_df_2], axis=0)
    compute_stats(total_df, all_preds, all_labels)
    total_df.to_csv(f"{tsv_name}.tsv", sep="\t", index=False)

    # with interjection
    my_examples_df, preds_1, labels_1 = evaluate_model(model, tokenizer, metaphor_set, middle_phrase=middle_phrase, use_cuda=use_cuda, verbose=verbose)
    mturk_examples_df, preds_2, labels_2 = evaluate_model(model, tokenizer, mturk_data.to_dict(orient="records"), middle_phrase=middle_phrase, use_cuda=use_cuda, verbose=verbose)
    mturk_examples_df_2, preds_3, labels_3 = evaluate_model(model, tokenizer, mturk_data_2.to_dict(orient="records"), middle_phrase=middle_phrase, use_cuda=use_cuda, verbose=verbose)
    
    all_preds = sum([preds_1, preds_2, preds_3], [])
    all_labels = sum([labels_1, labels_2, labels_3], [])
    total_df = pd.concat([my_examples_df, mturk_examples_df, mturk_examples_df_2], axis=0)
    compute_stats(total_df, all_preds, all_labels)
    total_df.to_csv(f"{tsv_name}_interject.tsv", sep="\t", index=False)
