import pandas as pd
from nltk import pos_tag, word_tokenize, RegexpParser

if __name__ == '__main__':
    df = pd.read_csv('mturk_xs.csv')
    df['noun'] = ''
    df['lastnoun'] = ''
    df['verb'] = ''
    df['adj'] = ''
    for index, row in df.iterrows():
        text = row['startphrase']
        pos_tags = pos_tag(word_tokenize(text))
        nouns = [word for word, tag in pos_tags if 'NN' in tag]
        verbs = [word for word, tag in pos_tags if 'V' in tag]
        adjs = [word for word, tag in pos_tags if 'JJ' in tag]
        if len(nouns) <= 1:
            print(text, row['valid_all3'])
            print(nouns)
            print(pos_tags)
        df.at[index, 'noun'] = nouns[0] if nouns else ''
        df.at[index, 'lastnoun'] = nouns[-1] if nouns else ''
        df.at[index, 'verb'] = verbs[0] if verbs else ''
        df.at[index, 'adj'] = adjs[0] if adjs else ''
        #chunker = RegexpParser(r'chunk:{<NN>+}')
        #print(chunker.parse(pos_tags))

    df.to_csv('pos_tagged.csv')



