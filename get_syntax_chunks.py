import pandas as pd
import spacy

def is_subarray(sub, arr):
    if len(arr) <= len(sub):
        return False

    for i in range(len(arr) - len(sub) + 1):
        if arr[i:i+len(sub)] == sub:
            return True
    return False

def remove_substrings(arr):
    '''Remove string from array that are a strict substring of another element'''
    to_delete = []
    for i in noun_phrases:
        for j in noun_phrases:
            if i != j and is_subarray(i.split(), j.split()):
                to_delete.append(i)
    return [x for x in noun_phrases if x not in to_delete]

def substring_between(string, a, b):
    a = a + ' '
    b = ' ' + b
    i = string.find(a)
    j = string.find(b)
    return string[i + len(a):j].strip()

if __name__ == '__main__':
    df = pd.read_csv('test.csv')
    df['x'] = ''
    df['y'] = ''
    df['z'] = ''

    nlp = spacy.load('en_core_web_sm')

    for index, row in df.iterrows():
        text = row['startphrase']
        doc = nlp(text)
        noun_phrases = []
        adj_phrases = []
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'pobj']:
                np = ' '.join([t.text for t in token.subtree])
                noun_phrases.append(np)
            elif token.dep_ in ['dobj', 'amod', 'acomp', 'advmod', 'attr']:
                ap = ' '.join([t.text for t in token.subtree])
                adj_phrases.append(ap)

        if len(noun_phrases) < 2:
            noun_phrases.extend(adj_phrases)

        if len(noun_phrases) > 2:
            noun_phrases_cleaned = remove_substrings(noun_phrases)
        else:
            noun_phrases_cleaned = noun_phrases


        if len(noun_phrases_cleaned) >= 2:
            x = noun_phrases_cleaned[0]
            z = noun_phrases_cleaned[-1]
            joined_text = ' '.join([word.text for word in doc]) # slightly different spaces from original text
            y = substring_between(joined_text, x, z)
            print(text)
            print(x,'|', y, '|',z)
            print()

            df.at[index, 'x'] = x
            df.at[index, 'y'] = y
            df.at[index, 'z'] = z

        else:
            print('FAILED PARSE:')
            print(text)
            print(len(noun_phrases), noun_phrases)
            print(len(adj_phrases), adj_phrases)
            print(len(noun_phrases_cleaned), noun_phrases_cleaned)
            print()





    df.to_csv('syntax_tagged.csv')



