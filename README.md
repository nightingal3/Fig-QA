# Testing the Ability of Language Models to Interpret Figurative Language

## Organization and Dataset
The main data splits are contained in `./data/filtered/`. Each is a CSV file which can be further processed into other formats. Each row of the CSV contains a `startphrase` (metaphorical context), `ending1` and `ending2`, as well as `labels`. Each pair of sentences has a `qid`.

```
./data/
├── commonsense_annotation/            # annotation of commonsense categories and lists of errors made by humans and models
├── dataset_characteristics/           # subject, relation, object splits for test set and medium train set        
├── filtered/
  ├── train_s.csv                      # small training set
  ├── train.csv                        # medium training set
  ├── train_xl.csv                     # large training set
  ├── dev.csv                          # validation set
  └── test.csv                         # test set
├── generation_annotation/             # annotation of GPT-3 Davinci's completions
├── human_responses/                   # human performance on the task (each file was completed by a different participant)
├── lm_train_data/                     # formatted train data to finetune models.    
├── prob_sheets/                       # probability score output from autoregressive models
├── common_metaphors.txt               # a list of common metaphors found online (https://leverageedu.com/blog/metaphors/).
└── random_words.txt                   # a list of random words used to prompt MTurk workers

```
## Usage
### To install the dependencies:

#### For conda users
```
conda env create -f environment.yml --name <env_name>
conda activate <env_name>
```

#### For pip users
`pip install -r requirements.txt`

### Zero-shot (scoring) for GPT-{2, neo}
```
python3 src/models/gpt_score.py {gpt2,gpt-neo-sm,gpt-neo-lg} \
[--middle_phrase=SUFFIX PROMPT] \
[--score_type={prob,loss}] \
[--use_prefix=N] \
[--verbose] \
[--cuda] \
[--multi_hop] \
[--out_file=PATH]
```
**middle_phrase**: The suffix prompt to use (we used "that is to say, ")

**score_type**: Return probability scores or LM loss in output. The predictions should be the same.

**use_prefix**: Number of random prefix (example) prompts to use.

**verbose**: Prints predictions 

**cuda**: use CUDA (you may have to install torch with CUDA enabled)

**multi-hop**: try multi-hop prediction. This was exploratory and not complete.

**out_file**: Write predictions (and probability scores) to this file. Defaults to `<model_id>_prob.csv.`

### Fine-tuning GPT-{2, neo}

### Zero-shot (scoring) for GPT-3

### Fine-tuning GPT-3

### Generation GPT-3

### Training BERT/RoBERTa

## Contact 

Insert contact information here afterwards
