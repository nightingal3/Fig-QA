import pandas as pd
import pdb
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
#stopwords = set(stopwords.words("english"))
determiners = ["the", "a", "an"]

def lemmatize_words(words):
    words = words.lower().translate(str.maketrans("", "", string.punctuation))

    return " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(words) if w not in determiners])

if __name__ == "__main__":
    segmented_file = "./dataset_characteristics/train_set_segmented.csv"
    df = pd.read_csv(segmented_file)
    df = df.dropna()
    
    subj = df["x"].apply(lemmatize_words)
    subj = subj.loc[subj.shift() != subj] # only count subj, obj etc once per pair

    rel = df["y"].apply(lemmatize_words)
    rel = rel.loc[rel.shift() != rel]

    obj = df["z"].apply(lemmatize_words)
    obj = obj.loc[obj.shift() != obj]

    subj_and_rel = pd.DataFrame({"subj": subj, "rel": rel})
    rel_and_obj = pd.DataFrame({"rel": rel, "obj": obj})
    subj_and_obj = pd.DataFrame({"subj": subj, "obj": obj})
    
    # todo: build co-occurrence matrix by hand
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
    