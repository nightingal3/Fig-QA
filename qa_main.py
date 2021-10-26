import transformers
from transformers import AutoTokenizer, BertTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, BertModel, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
import numpy as np
import pdb
from dataclasses import dataclass
from typing import Optional, Union
from sample_metaphors import trial_dataset
import torch
# A lot of the boilerplate is from this notebook on setting up BERT for QA: 
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/multiple_choice.ipynb

bert = "bert-base-uncased"
gpt_neo = "EleutherAI/gpt-neo-1.3B"
roberta = "roberta-base"
batch_size = 16

def preprocess_examples(examples_df, tokenizer) -> dict:
    context_sentence = [[context] * 2 for context in examples_df["startphrase"]] 
    context_flat = sum(context_sentence, [])
    second_sentence = [[ending1, ending2] for ending1, ending2 in zip(examples_df["ending1"], examples_df["ending2"])]
    second_sent_flat = sum(second_sentence, [])
    tokenized_examples = tokenizer(context_flat, second_sent_flat, return_tensors='pt', padding=True)
    
    return {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}


def preprocess_function_dataset(examples):
    ending_names = ["ending1", "ending2"]
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 2 for context in examples["startphrase"]]
    # Grab all second sentences possible for each context.
    #second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    second_sentence = [[ending1, ending2] for ending1, ending2 in zip(examples["ending1"], examples["ending2"])]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentence, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def evaluate_model(model, test_data, labels) -> float:
    model.eval()
    preds = []
    for input_id, attn_mask, token_type_id in zip(test_data["input_ids"], test_data["attention_mask"], test_data["token_type_ids"]):
        input_id, attn_mask, token_type_id = input_id.unsqueeze(0), attn_mask.unsqueeze(0), token_type_id.unsqueeze(0)
        output = model(input_ids=input_id, attention_mask=attn_mask, token_type_ids=token_type_id)
        pred = torch.argmax(output["logits"]).item()
        preds.append(pred)

    return len(np.where(np.array(preds) == np.array(labels))[0])/len(labels), preds

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def display_predictions(preds, labels, df_test):
    for i, (pred, label) in enumerate(zip(preds, labels)):
        status = "CORRECT" if pred == label else "WRONG"
        startphrase = df_test.iloc[i]["startphrase"]
        option1 = df_test.iloc[i]["ending1"]
        option2 = df_test.iloc[i]["ending2"]
        model_pred = option1 if pred == 0 else option2
        print(f"=== Number {i}, got it {status} ===")
        print(f"{startphrase}")
        print(f"1.{option1}")
        print(f"2.{option2}")
        print(f"Prediction: {model_pred}")

if __name__ == "__main__":
    mode = "test"
    selected_model = bert

    trial_df_test = pd.DataFrame(trial_dataset["test"])
    my_dataset = load_dataset("csv", data_files={"train": "train_data_mturk.csv", "validation": "train_data_mturk.csv"})

    #labels_train = list(trial_df["train"]["label"])
    labels_test = list(trial_df_test["label"])
    tokenizer = AutoTokenizer.from_pretrained(selected_model, use_fast=True)
    neo_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    features_test = preprocess_examples(trial_df_test, tokenizer)
    #features_test = preprocess_examples(trial_df["test"], tokenizer_bert)
    #print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])
    #print(features.keys())
    #idx = 0
    #print([tokenizer.decode(features["input_ids"][idx][i]) for i in range(2)])
    #accepted_keys = ["input_ids", "attention_mask", "label"]
    #features_new = {k: v for k, v in features.items() if k in accepted_keys}
    #batch = DataCollatorForMultipleChoice(tokenizer)(features_new) 

    # FILLER: train with other dataset for now because data not collected
    #datasets = load_dataset("swag", "regular")
    #encoded_datasets = datasets.map(preprocess_function_swag, batched=True)
    if mode == "train":
        encoded_datasets = my_dataset.map(preprocess_function_dataset, batched=True)
        #bert_model = AutoModelForMultipleChoice.from_pretrained(bert_model)
        model = AutoModelForMultipleChoice.from_pretrained(selected_model)
        args = TrainingArguments(
            "testing-qa",
            evaluation_strategy = "epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=True,
        )
        trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
        )
        trainer.train()
        model.save_pretrained(f"./testing-qa/{selected_model}-few-shot/")

    if mode == "test":
        model = AutoModelForMultipleChoice.from_pretrained(f"./testing-qa/{selected_model}-few-shot/")
        acc, preds = evaluate_model(model, features_test, labels_test)
        print(acc)
        print(preds)
        display_predictions(preds, labels_test, trial_df_test)

