# Testing the Ability of Language Models to Interpret Figurative Language

## Table of Contents
* [Introduction](#introduction)
* [Organization and Dataset](#organization-and-dataset)
* [Usage: Evaluate your own models](#usage-evaluate-your-own-models)
* [Usage: Reproducing figures from the paper](#usage-reproducing-figures)
* [Contact](#contact)
* [Citation](#citation)

## Introduction
This repository contains the dataset and code for the paper [Testing the Ability of Language Models to Interpret Figurative Language](arxiv-link-here). Fig-QA consists of 10256 examples of human-written creative metaphors that are paired as a Winograd schema. It can be used to evaluate the commonsense reasoning of models. The metaphors themselves can also be used as training data for other tasks, such as metaphor detection or generation. 

#### Why Figurative language?
Most of NLP (as of the publication date of this paper) focuses on literal interpretation of phrases. However, this isn't the only way in which humans use language. In most cases, people can readily interpret creative phrases such as "She thinks of herself as a particle of sand in the desert", even if they have not directly heard such a phrase before. Figurative language is prominent in colloquial text and literature, and correct inference regarding figurative language involves commonsense knowledge as well as flexibility in word meaning inference.

#### Examples
The dataset is formatted as a Winograd schema. This means that sentences with the same beginning, but opposite meaning are paired together. This formatting was designed to reduce shortcut learning. Accuracy is calculated over all examples though, rather than for pairs.

| Sentence  | Correct Answer |
| ------------- | ------------- |
| The future is as bright as the sun  | The future is bright  |
| The future is as bright as ink  | The future is not bright  |
| The concert was as crowded as a rush-hour train  | The concert was crowded  |
| The concert was as crowded as a mausoleum  | The concert was not crowded  |
| Sleeping over at his house is like spending a night at the Waldorf Astoria | He has very nice accommodations |
| Sleeping over at his house is like spending a night at the Motel 6 | He has below average accommodations |

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
## Usage (evaluate your own models)
### Offline evaluation

You can evaluate your models on the [dev set](https://github.com/nightingal3/metaphor-qa/blob/master/data/filtered/dev.csv). The labels for the test set are hidden, but you can still see the questions [here](https://github.com/nightingal3/metaphor-qa/blob/master/data/filtered/test.csv).  

### Submitting systems to Explainaboard

You can evaluate your models on the test set by submitting to the [leaderboard](https://explainaboard.inspiredco.ai/leaderboards?dataset=fig_qa) on Explainaboard. Click on "New" and select `qa-multiple-choice` for the task field. Select `accuracy` for the metric. You should upload results in the form of a system output file in JSON or JSONL format. 

#### System output format
The system output format is detailed [here](https://github.com/neulab/ExplainaBoard/blob/753e98011e41337eaddaa0546a50e5b6d4747e87/docs/task_qa_multiple_choice.md).

#### More about Explainaboard

You can learn more about Explainaboard [here](https://github.com/neulab/ExplainaBoard). 

## Usage (reproducing figures)

Note: since the test set is now hidden, these results are for the dev set, rather than for the test set as reported in the paper. You can see model performance for the test set on Explainaboard. 

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
* **middle_phrase**: The suffix prompt to use (we used "that is to say, ")

* **score_type**: Return probability scores or LM loss in output. The predictions should be the same.

* **use_prefix**: Number of random prefix (example) prompts to use.

* **verbose**: Prints predictions 

* **cuda**: use CUDA (you may have to install torch with CUDA enabled)

* **multi-hop**: try multi-hop prediction. This was exploratory and not complete.

* **out_file**: Write predictions (and probability scores) to this file. Defaults to `<model_id>_prob.csv.`

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

* **dont_train**: skip training and just evaluate

* **dont_eval**: only train and don't eval

* **train_path**: path to processed train file (defaults to `"./data/lm_train_data/train.txt"`)

* **eval_path**: path to processed validation file (defaults to `"./data/lm_train_data/dev.txt"`)

* **seed**: random seed. Defaults to 42

* **cuda**: use CUDA (you may have to install torch with CUDA enabled)

* **num_epochs**: number of epochs to train for. Defaults to 3. Overridden by early stopping.

* **learning_rate**: learning rate. Defaults to 5e-5.

* **middle_phrase**: The suffix prompt to use (we used "that is to say, ")

* **prefix**: Number of random prefix (example) prompts to use.

* **contrastive**: use contrastive training or not. Did not work for GPT-* models in this project.

* **contrast_lambd**: also depreciated, hyperparameter for contrastive train.

* **log_history**: log eval loss at each epoch.

* **deepspeed**: use deepspeed. Required for GPT-neo.

* **out_file**: Write predictions (and probability scores) to this directory. Defaults to `"./experiments/<model_name>/epochs_<num_epochs>_<learning_rate>_/seed_<seed>"`.

* **early_stopping**: use early stopping.


### GPT-3 Zero-shot, Finetuning and Generation
Code to call the OpenAI API are in the following notebooks:
- Zero-shot (scoring): `./src/notebooks/OpenAI Probabilities.ipynb`.
- Finetuning: `./src/notebooks/OpenAI Finetune.ipynb`.
- Completion generation: `./src/notebooks/completions.ipynb`.

Please use your own API key.

### Training BERT/RoBERTa
We use the code from [WinoGrande](https://github.com/allenai/winogrande) without modification.
Please use `./src/winogrande/preproc_to_winogrande_format.py` to preprocess the data into the WinoGrande codebase 
format, and follow the instructions in that repository. Our sample run scripts are included in `./src/winogrande/*.sh`

### Producing the figures

Visualization scripts are in `./src/visualizations/`.

#### Figure 1 (dataset visualization)
```
python3 src/visualizations/dataset_vis.py
```
Saves by default in subj_bar.png, obj_bar.png, rel_bar.png.
Does not run by default, but hypernyms/POS of each segment can be found with `get_hypernyms`/`get_pos_tags`.

#### Figure 2 (prompting)
```
python3 src/visualizations/make_performance_bar_chart.py
```
This will generate two figures, one for zero-shot and finetuning performance and one for prompting performance, however only the second was used in the paper (the first figure's data was presented in Table 6 instead).

#### Figure 3 (odds and probability)
```
python3 src/visualizations/plot_log_odds_vs_pred.py
```
Running this will produce the plots trained_prob.png and untrained_prob.png. Spearman R values (Table 5) will also be printed.

## Contact 

Insert contact information here afterwards

## Citation 

Insert citation here afterwards
