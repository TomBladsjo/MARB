# MARB 
### – a dataset for studying Marked Attribute Reporting Bias

This repository contains the code used for dataset creation and model evaluation in the Master's thesis "Don't Mention the Norm – On the Relationship Between Reporting Bias and Social Bias in Humans and Language Models" (`thesis.pdf`).

### About

Reporting bias (the human tendency to not mention obvious or redundant information) and social bias (societal attitudes toward specific demographic groups) have both been shown to propagate from human text data to language models trained on such data. 
However, the two phenomena have not previously been studied in combination. 
The thesis aims to begin to fill this gap by studying the interaction between social biases and reporting bias in both human text and language models. 
To this end, the MARB dataset was developed. 
Unlike many existing benchmark datasets, MARB does not rely on artificially constructed templates or crowdworkers to create contrasting examples. 
Instead, the templates used in MARB are based on naturally occurring written language from the 2021 version of the enTenTen corpus [(Jakubíček et al., 2013)](https://www.sketchengine.eu/ententen-english-corpus/). 

### Dataset

The dataset consists of nearly 30K template sequences – 9.5K containing each of the phrases "a person", "a woman", "a man" – and their modifications.
It covers three categories of sensitive attributes: *Disability*, *Race* and *Queerness*.
Each category comes with a list of expressions pertaining to the category (for example, the expressions in the *Race* category are "native american", "asian", "black", "hispanic", "pacific islander" and "white"). 
Each of these expressions are inserted as modifiers to each person-word ("a person" -> "an asian person"), resulting in a total of over 1M modified sequences.
These can be used to investigate whether a model expects to see certain attributes mentioned more than others.

### Usage

##### Dataset

The full dataset can be found at `data/data.tar.gz`. 
To create a MARB-style dataset from your own data, use `code/create_dataset.py`.

##### Model evaluation

To evaluate a model on the dataset, run:

`python code/test_models.py <model> <path to dataset directory> <path to directory for result files>`

where `model` is the model name as a string (case insensitive).
Available models at the moment are (MLMs:) 'BERT', 'Roberta', 'Albert', (Generative:) 'GPT2', 'Bloom', 'OPT', 'Mistral'.
To see all available options, run
`python code/test_models.py -h`.

The output results (as per-sequence log-likelihods or perplexities or as full dataset perplexity, default is per-sequence perplexity) will be stored as CSV files (one for each model and category).
