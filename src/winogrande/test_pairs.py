import pandas as pd

# this script computes stats for predictions on paired/unpaired data
if __name__ == '__main__':
    df = pd.read_csv('dev_preds.csv')
    pairs = []
    pairs_single = []
    pairs_wrong = []
    pairs_correct = []
    singles = []
    singles_correct = []
    i = 0
    while i < len(df) - 1:
        if df['sentence'][i][:5] == df['sentence'][i+1][:5]:
            p1 = df['answer==pred'][i]
            p2 = df['answer==pred'][i+1]
            pairs.append(i)
            if not p1 and not p2:
                pairs_wrong.append(i)
            elif not p1 or not p2:
                pairs_single.append(i)
            else:
                pairs_correct.append(i)
            i += 1
        else:
            if df['answer==pred'][i]:
                singles_correct.append(i)
            singles.append(i)
        i += 1
    singles.append(len(df))

    print('pairs', len(pairs))
    print('pair_correct', len(pairs_correct))
    print('pair_singles', len(pairs_single))
    print('pair_doubles', len(pairs_wrong))
    assert len(pairs) == len(pairs_single) + len(pairs_wrong) + len(pairs_correct)
    print('singles', len(singles))
    print('singles_correct', len(singles_correct))
    assert len(singles) + 2 * len(pairs) == len(df)

