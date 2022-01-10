import pandas as pd
import pdb
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

lemmatizer = WordNetLemmatizer()
#stopwords = set(stopwords.words("english"))
determiners = ["the", "a", "an", "that"]

sns.set(font_scale=1.5)

def lemmatize_words(words):
    words = words.lower().translate(str.maketrans("", "", string.punctuation))

    return " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(words) if w not in determiners])

def make_count_barplot(df: pd.DataFrame, out_filename: str, x_label: str, palette_name: str = "flare") -> None:
    plt.gca().clear()
    ax = sns.countplot(data=df, x="subj", order=df["subj"].value_counts().index, palette=palette_name)
    plt.xlim(0, 24)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title(x_label)
    plt.tight_layout()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")

def get_hypernyms(sent: str, wordnet_pos: List) -> List:
    hypernyms = []
    #print(sent)
    for word, pos in zip(sent.split(), wordnet_pos):
        ignore = ["s", "at", "m", "in"]
        if word in ignore or (len(word) == 1 and word != "i"):
            continue
        # common errors
        if word == "he":
            word = "man"
        if word == "she":
            word = "woman"
        if word == "i":
            word = "me"
            
        if pos != "other":
            synset = wordnet.synsets(word, pos=pos)
        else:
            synset = wordnet.synsets(word)
        #for ss in synset:
            #print(word)
            #print(ss)
            #print(ss.hypernyms())
        if len(synset) != 0 and len(synset[0].hypernyms()) != 0:
            hypernym_names = [ss.name() for ss in synset[0].hypernyms()]
            hypernyms.extend(hypernym_names)
    #print(hypernyms)
    return hypernyms
    
def get_pos_tags(sent: str) -> List:
    return " ".join([x[1] for x in nltk.pos_tag(word_tokenize(sent))])

def get_wordnet_pos(treebank_tags: str) -> List:
    wordnet_tags = []
    for treebank_tag in treebank_tags.split():
        if treebank_tag.startswith('J'):
            wordnet_tags.append(wordnet.ADJ)
        elif treebank_tag.startswith('V'):
            wordnet_tags.append(wordnet.VERB)
        elif treebank_tag.startswith('N'):
            wordnet_tags.append(wordnet.NOUN)
        elif treebank_tag.startswith('R'):
            wordnet_tags.append(wordnet.ADV)
        else:
            wordnet_tags.append("other")
    return wordnet_tags

def correctness_by_pos(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    counts = df["pos_tag"].value_counts()
    df = df[df['pos_tag'].isin(counts[counts >= threshold].index)]
    tag_correctness = {}
    for tag in df["pos_tag"].unique():
        tag_correctness[tag] = len(df.loc[(df["pos_tag"] == tag) & (df["correctness"] == True)])/len(df.loc[df["pos_tag"] == tag])

    return tag_correctness, dict(counts)

def correctness_by_hypernym(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    a = pd.Series([item for sublist in df["hypernyms"] for item in sublist])
    counts = dict(a.value_counts())
    selection = {key: item for key, item in counts.items() if item >= 10}
    mask = df["hypernyms"].apply(lambda x: any(item for item in selection if item in x))
    df = df[mask]

    hyp_correctness = {}
    for hypernym in selection:
        selected_rows = df[df["hypernyms"].apply(lambda x: hypernym in x)]
        hyp_correctness[hypernym] = len(selected_rows.loc[selected_rows["correctness"] == True])/len(selected_rows)
    
    return hyp_correctness, counts

if __name__ == "__main__":
    segmented_file = "./dataset_characteristics/test_segmented.csv"
    errors_file = "./gpt3_incorrect.csv"
    df = pd.read_csv(segmented_file)
    df = df.dropna()
    errors = set(list(pd.read_csv(errors_file)["startphrase"]))

    subj = df["x"].apply(lemmatize_words)
    #subj = subj.loc[subj.shift() != subj] # only count subj, obj etc once per pair
    dummy_sub = pd.DataFrame({"subj": subj, "startphrase": df["startphrase"]})

    rel = df["y"].apply(lemmatize_words)
    #rel = rel.loc[rel.shift() != rel]
    rel = rel.str.replace("wa\\b", "was", regex=True)
    rel = rel.str.replace("a\\b", "as", regex=True)
    rel = rel.str.replace("ha\\b", "has", regex=True)
    dummy_rel = pd.DataFrame({"subj": rel, "startphrase": df["startphrase"]})

    obj = df["z"].apply(lemmatize_words)
    #obj = obj.loc[obj.shift() != obj]
    dummy_obj = pd.DataFrame({"subj": obj, "startphrase": df["startphrase"]})

    subj_unique = len(subj.value_counts())
    rel_unique = len(rel.value_counts())
    obj_unique = len(obj.value_counts())

    dummy_sub["pos_tag"] = dummy_sub["subj"].apply(get_pos_tags)
    dummy_sub["wordnet_pos"] = dummy_sub["pos_tag"].apply(get_wordnet_pos)
    dummy_sub["hypernyms"] = [get_hypernyms(sent, pos) for sent, pos in zip(dummy_sub["subj"], dummy_sub["wordnet_pos"])]
    dummy_sub["correctness"] = dummy_sub["startphrase"].apply(lambda x: x not in errors)
    correctness_sub, counts_sub = correctness_by_pos(dummy_sub)
    correctness_hyp_sub, counts_hyp_sub = correctness_by_hypernym(dummy_sub)
    #print(sorted(correctness_sub.items(), key=lambda x: counts_sub[x[0]], reverse=True))
    #print(sorted(correctness_hyp_sub.items(), key=lambda x: counts_hyp_sub[x[0]], reverse=True))
    #pdb.set_trace()
    dummy_rel["pos_tag"] = dummy_rel["subj"].apply(get_pos_tags)
    dummy_rel["wordnet_pos"] = dummy_rel["pos_tag"].apply(get_wordnet_pos)
    dummy_rel["hypernyms"] = [get_hypernyms(sent, pos) for sent, pos in zip(dummy_rel["subj"], dummy_rel["wordnet_pos"])]
    dummy_rel["correctness"] = dummy_rel["startphrase"].apply(lambda x: x not in errors)

    #print(sorted(correctness_rel.items(), key=lambda x: counts_rel[x[0]], reverse=True))
    

    dummy_obj["pos_tag"] = dummy_obj["subj"].apply(get_pos_tags)
    dummy_obj["wordnet_pos"] = dummy_obj["pos_tag"].apply(get_wordnet_pos)
    dummy_obj["hypernyms"] = [get_hypernyms(sent, pos) for sent, pos in zip(dummy_obj["subj"], dummy_obj["wordnet_pos"])]
    dummy_obj["correctness"] = dummy_obj["startphrase"].apply(lambda x: x not in errors)
    correctness_obj, counts_objs = correctness_by_pos(dummy_obj)
    correctness_hyp_obj, counts_hyp_obj = correctness_by_hypernym(dummy_obj)

    #print(sorted(correctness_obj.items(), key=lambda x: counts_objs[x[0]], reverse=True))
    #print(counts_objs)
    print(sorted(correctness_hyp_obj.items(), key=lambda x: counts_hyp_obj[x[0]], reverse=True))
    print(counts_hyp_obj)
    #rel = rel.value_counts().rename_axis('unique_values').reset_index(name='counts')
    pdb.set_trace()
    make_count_barplot(dummy_sub, "subj_bar", "Subject", "flare")
    make_count_barplot(dummy_rel, "rel_bar", "Relation", "crest")
    make_count_barplot(dummy_obj, "obj_bar", "Object", "viridis")

    print("unique subjects: ", subj_unique)
    print("unique relations: ", rel_unique)
    print("unique objects: ", obj_unique)
    assert False
    subj_and_rel = pd.DataFrame({"subj": subj, "rel": rel})
    rel_and_obj = pd.DataFrame({"rel": rel, "obj": obj})
    subj_and_obj = pd.DataFrame({"subj": subj, "obj": obj})
    
    subj_rel_cooccurrence = np.zeros((subj_unique, rel_unique))
    # todo: build co-occurrence matrix by hand
    for s in subj_and_rel["subj"]:
        contains_subj = subj_and_rel.loc[subj_and_rel["subj"] == s]
        pdb.set_trace()
    s_o = subj_and_rel.stack().str.get_dummies().sum(level=0).ne(0).astype(int)
    s_o =s_o.T.dot(s_o).astype(float)
    np.fill_diagonal(s_o.values, np.nan)

    a = s_o.stack()   
    a = a[a >= 1].rename_axis(('source', 'target')).reset_index(name='weight')
    pdb.set_trace()
    G = nx.from_pandas_edgelist(a, edge_attr=True)
    X, Y = nx.bipartite.sets(G)
    pos = nx.bipartite_layout(G, X)
    nx.draw_networkx(G, with_labels=True, pos=pos, width=edge_widths * 5)
    plt.show()
    