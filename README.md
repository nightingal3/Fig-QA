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

### Zero-shot (scoring) for GPT-{2, neo}
```
python3 src/models/gpt_score.py {gpt2,gpt-neo-sm,gpt-neo-lg} \
[--middle_phrase=SUFFIX_PROMPT] \
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
```
python3 src/models/train_lm_models.py {gpt2,gpt-neo-sm,gpt-neo-lg} \
[--dont_train] \
[--dont_eval] \
[--train_path=TRAIN_PATH] \
[--eval_path=EVAL_PATH] \
[--seed=SEED] \
[--cuda] \
[--num_epochs=NUM_EPOCHS] \
[--learning_rate=LR] \
[--middle_phrase=SUFFIX_PROMPT] \
[--prefix=N] \
[--contrastive] \
[--contrast_lambd=a] \
[--log_history] \
[--deepspeed] \
[----out_path=PATH] \
[----early_stopping]
```

**dont_train**: skip training and just evaluate

**dont_eval**: only train and don't eval

**train_path**: path to processed train file (defaults to `"./data/lm_train_data/train.txt"`)

**eval_path**: path to processed validation file (defaults to `"./data/lm_train_data/dev.txt"`)

**seed**: random seed. Defaults to 42

**cuda**: use CUDA (you may have to install torch with CUDA enabled)

**num_epochs**: number of epochs to train for. Defaults to 3. Overridden by early stopping.

**learning_rate**: learning rate. Defaults to 5e-5.

**middle_phrase**: The suffix prompt to use (we used "that is to say, ")

**prefix**: Number of random prefix (example) prompts to use.

**contrastive**: use contrastive training or not. Did not work for GPT-* models in this project.

**contrast_lambd**: also depreciated, hyperparameter for contrastive train.

**log_history**: log eval loss at each epoch.

**deepspeed**: use deepspeed. Required for GPT-neo.

**out_file**: Write predictions (and probability scores) to this directory. Defaults to `"./experiments/<model_name>/epochs_<num_epochs>_<learning_rate>_/seed_<seed>"`.

**early_stopping**: use early stopping.

****
### Zero-shot (scoring) for GPT-3

### Fine-tuning GPT-3

### Generation GPT-3

### Training BERT/RoBERTa

## Contact 

Insert contact information here afterwards
