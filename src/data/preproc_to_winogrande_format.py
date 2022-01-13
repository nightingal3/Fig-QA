import pandas as pd
from tqdm import tqdm

def proc(sent):
    if not sent.endswith("."):  # finish with period
        sent += '.'
    if not sent[0].isupper():  # start with a capital letter
        sent = sent[0].upper() + sent[1:]
    # escape quotation characters
    return sent.replace('"', '\\"')


def preproc_to_winogrande_format(df, out_path):
    out_lines = []
    line_format = '"qID": "{qID}", "sentence": "{sentence}", "option1": "{option1}", "option2": "{option2}", "answer": "{answer}"'
    for i, line in tqdm(df.iterrows(), total=df.shape[0]):
        s = proc(line['startphrase']) + ' _'
        o1 = proc(line['ending1'])
        o2 = proc(line['ending2'])
        out_lines.append(
            '{' + line_format.format(qID=line.get('qid', i), sentence=s, option1=o1, option2=o2, answer=int(line['labels'])+1) + '}\n'
        )
    with open(out_path, 'w') as f:
        f.writelines(out_lines)


if __name__ == '__main__':
    for split in ('train_xl', 'train_m'):
        df = pd.read_csv(f"/home/cuichenx/Courses/11-711/A4/data/{split}.csv")
        df = df[df.valid == 1]
        out_lines = f"../data/metaphor_{split}.jsonl"
        preproc_to_winogrande_format(df, out_lines)
