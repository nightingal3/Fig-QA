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
        ctx, p1, p2 = metaphor_data["startphrase"], metaphor_data["ending1"], metaphor_data["ending2"]        
        sent1 = ctx + ". " + middle_phrase + p1 + "."
        sent2 = ctx + ". " + middle_phrase + p2 + "."
        if not os.path.isdir(f"./attn_test/{i}"):
            os.mkdir(f"./attn_test/{i}")

        _, attn1, input_ids_1 = sent_scoring((model, tokenizer), sent1, use_cuda, score_type, output_attentions=True)
        attn_avg_1 = torch.mean(attn1[-1], dim=1).squeeze(0)

        _, attn2, input_ids_2 = sent_scoring((model, tokenizer), sent2, use_cuda, score_type, output_attentions=True)
        attn_avg_2 = torch.mean(attn2[-1], dim=1).squeeze(0)

        first_mismatch = np.where(np.array(input_ids_1[0]) != np.array(input_ids_2[0]))[0][0]
        for j in range(first_mismatch, min(len(attn_avg_1), len(attn_avg_2))):
            if len(attn_avg_1[j]) != len(attn_avg_2[j]):
                pdb.set_trace()
            subtracted_attn = (attn_avg_1[j][:first_mismatch] - attn_avg_2[j][:first_mismatch])
            visualize(subtracted_attn, tokenizer, input_ids_1[:, :first_mismatch], normalize=False, save_file=f"{i}_test_{j}.png")

        if i == 3:
            assert False
            
def plot_attention_individual(model, tokenizer, dataset, out_dir):
    pass

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
