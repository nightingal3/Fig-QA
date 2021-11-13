import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import torch
import numpy as np
import os

from gpt_score import model_init, sent_scoring
from sample_metaphors import trial_dataset

plt.rcParams['font.size'] = '60'

# This is Kayo's visualization function for attention on the input sequence
def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, save_file=None, title=None, figsize=60):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
        attention = np.array([attention])
        attn_min, attn_max = -1, 1
    else:
        attn_min, attn_max = min(attention).item(), max(attention).item()
        attention = np.expand_dims(np.array(attention), axis=0)

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=attn_min, vmax=attn_max)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.3e}'.format(z), ha='center', va='center', fontsize=30)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()

def plot_contrastive_attention(model, tokenizer, dataset): 
    for i, metaphor_data in enumerate(dataset):
        ctx, p1, p2, label = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"], metaphor_data["label"]      
        sent1 = ctx + ". " + middle_phrase + p1 + "."
        sent2 = ctx + ". " + middle_phrase + p2 + "."
        if not os.path.isdir(f"./attn_test/{i}"):
            os.mkdir(f"./attn_test/{i}")

        _, attn1, input_ids_1 = sent_scoring((model, tokenizer), sent1, use_cuda, score_type, output_attentions=True)
        attn_avg_1 = torch.mean(attn1[-1], dim=1).squeeze(0)

        _, attn2, input_ids_2 = sent_scoring((model, tokenizer), sent2, use_cuda, score_type, output_attentions=True)
        attn_avg_2 = torch.mean(attn2[-1], dim=1).squeeze(0)

        min_len = min(len(input_ids_1[0]), len(input_ids_2[0]))
        truncated_1 = input_ids_1[0][:min_len]
        truncated_2 = input_ids_2[0][:min_len]
        correct_answer = attn_avg_1 if label == 0 else attn_avg_2
        wrong_answer = attn_avg_1 if label == 1 else attn_avg_2
        correct_ids = input_ids_1 if label == 0 else input_ids_2
        wrong_ids = input_ids_1 if label == 1 else input_ids_2

        correct_words = [tokenizer.decode(i) for i in correct_ids[0]]
        wrong_words = [tokenizer.decode(i) for i in wrong_ids[0]]

        # fix this for when the sentences are the same length
        first_mismatch = np.where(np.array(truncated_1) != np.array(truncated_2))[0][0]
        for j in range(first_mismatch, min(len(attn_avg_1), len(attn_avg_2))):
            subtracted_attn = (correct_answer[j][:first_mismatch] - wrong_answer[j][:first_mismatch])
            visualize(subtracted_attn, tokenizer, input_ids_1[:, :first_mismatch], normalize=False, save_file=f"./attn_test/{i}/{j}_correct_{correct_words[j]}_wrong_{wrong_words[j]}.png")
            
def plot_attention_individual(model, tokenizer, dataset, out_dir):
    pass

# TODO: test token alignment
# adapted from this notebook: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=GL91xMGP9_CO

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    use_cuda = False
    model_names = {"gpt2": "gpt2", "neo-sm": "EleutherAI/gpt-neo-1.3B", "neo-lg": "EleutherAI/gpt-neo-2.7B"}
    model_id = "gpt2"
    model_name = model_names[model_id]
    tsv_name = f"{model_id}_prob"
    middle_phrase = "That is to say, "
    score_type = "prob"
    verbose=False
        
    metaphor_set = trial_dataset["test"]
    model, tokenizer = model_init(model_name, use_cuda, output_attentions=True)

    plot_contrastive_attention(model, tokenizer, metaphor_set)
    assert False

    for i, metaphor_data in enumerate(metaphor_set):
        ctx, p1, p2 = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"]        
        sent1 = ctx + ". " + middle_phrase + p1 + "."
        sent2 = ctx + ". " + middle_phrase + p2 + "."
        if not os.path.isdir(f"./attn_test/{i}"):
            os.mkdir(f"./attn_test/{i}")

        for sent_no, sent in enumerate([sent1, sent2]):
            _, attn, input_ids = sent_scoring((model, tokenizer), sent, use_cuda, score_type, output_attentions=True)
            attn_avg = torch.mean(attn[-1], dim=1).squeeze(0)
            words = [tokenizer.decode(i) for i in input_ids[0][:len(attn_avg) + 1]]
            for j in range(len(attn_avg)):
                visualize(attn_avg[j], tokenizer, input_ids, normalize=True, save_file=f"attn_test/{i}/sent_{sent_no}_word_{j}_{words[j]}.png", title=words[j])
