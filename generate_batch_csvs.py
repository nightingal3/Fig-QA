import random
import csv

if __name__ == '__main__':
    start_index = 18
    end_index = 100
    words = []
    with open('random_words.txt') as f:
        words = [line.strip() for line in f]

    random.shuffle(words)

    i = 0
    while start_index <= end_index:
        with open(f'batch_{start_index}.csv', mode='w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=['w1','w2','w3'])
            writer.writeheader()
            writer.writerow({'w1': words[i], 'w2': words[i+1], 'w3': words[i+2]})
        i += 3
        start_index += 1




